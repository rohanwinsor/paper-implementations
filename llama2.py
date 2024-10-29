import torch
import os
import torch.nn as nn
from dataclasses import dataclass
from utils.utils import model_memory_size
from typing import TypeAlias
import sentencepiece as spm
from huggingface_hub import login
from utils.utils import generate, text_to_token_ids, token_ids_to_text, load_weights_into_llama
try:
    login(token=os.environ["HF_ACCESS_TOKEN"])
except:
    print("ERROR :: UNABLE TO LOGIN")
Tensor: TypeAlias = torch.Tensor


class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)

@dataclass
class Llama2Config:
    vocab_size = 32000
    context_length = 4096
    emb_dim = 4096
    n_heads = 32
    n_layers = 32
    hidden_dim = 11008
    dtype = torch.bfloat16


class RMSNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(embed_dim)).float()

    def forward(self, x: torch.Tensor):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed: torch.Tensor = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.fc1 = nn.Linear(
            config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False
        )
        self.fc2 = nn.Linear(
            config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False
        )
        self.fc3 = nn.Linear(
            config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False
        )
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


# YANKED THIS CODE from here -> https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, n_head, dtype=None):
        assert d_out % n_head == 0
        super().__init__()
        self.W_keys = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_queries = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_values = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = d_out // n_head
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )
        cos, sin = precompute_rope_params(self.head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.shape
        keys: Tensor = self.W_keys(x).view(B, T, self.n_head, self.head_dim)
        queries: Tensor = self.W_queries(x).view(B, T, self.n_head, self.head_dim)
        values: Tensor = self.W_values(x).view(B, T, self.n_head, self.head_dim)
        ## (B, T, self.n_head, self.head_dim) -> (B, self.n_head, T,  self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)
        omega = queries @ keys.transpose(2, 3)
        # mask
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        # scaled att weight
        att_weight = torch.softmax(omega / keys.shape[-1] ** 0.5, dim=-1)
        # (B, no_of_heads, T, head_dim) -> (B, T, no_of_heads, head_dim)
        out = (att_weight @ values).transpose(1, 2)
        return self.out_proj(out.contiguous().view(B, T, self.d_out))


class TransformerBlock(nn.Module):
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.att = MultiHeadAttention(
            config.emb_dim,
            config.emb_dim,
            config.context_length,
            config.n_heads,
            config.dtype,
        )
        self.ff = FeedForward(config)
        self.rms_norm1 = RMSNorm(config.emb_dim)
        self.rms_norm2 = RMSNorm(config.emb_dim)

    def forward(self, x):
        shortcut = x
        x = self.rms_norm1(x)
        x = self.att(x)
        x = x + shortcut
        shortcut = x
        x = self.ff(self.rms_norm2(x))
        x = x + shortcut
        return x


class LLama2(nn.Module):
    ## embedding
    ## Transformers Block * n_layers
    ## RMS
    ## Linear
    def __init__(self, config: Llama2Config):
        super().__init__()
        self.tok_embed = nn.Embedding(
            config.vocab_size, config.emb_dim, dtype=config.dtype
        )
        self.transformers_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.emb_dim)
        self.output = nn.Linear(
            config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype
        )

    def forward(self, x):
        tok_emb = self.tok_embed(x)
        x = self.transformers_block(tok_emb)
        x = self.norm(x)
        return self.output(x)


if __name__ == "__main__":
    model = LLama2(Llama2Config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print(
        f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB"
    )
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
    from huggingface_hub import hf_hub_download

    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="tokenizers/Llama-2-7b"
    )
    tokenizer = LlamaTokenizer(tokenizer_file)
    device = "mps"

    weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",
   filename="consolidated.00.pth",
   local_dir="models/Llama-2-7b"
)
    weights = torch.load(weights_file, weights_only=True)

    load_weights_into_llama(model, Llama2Config, weights)
    model.to(device)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
        max_output_token=5,
        context_length=Llama2Config().context_length,
        top_k=1,
        temperate=0.
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
