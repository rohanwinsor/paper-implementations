import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor
### FeedForward Layer
### LayerNorm
### MultiHead Attention
### GeLU
### Transformer Block
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, n_head, qkv_bias=False):
        assert d_out % n_head == 0
        super().__init__()
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = d_out // n_head
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        B, T, C = x.shape
        keys: Tensor = self.W_keys(x).view(B, T, self.n_head, self.head_dim)
        queries: Tensor = self.W_queries(x).view(B, T, self.n_head, self.head_dim)
        values: Tensor = self.W_values(x).view(B, T, self.n_head, self.head_dim)
        ## (B, T, self.n_head, self.head_dim) -> (B, self.n_head, T,  self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)        
        values = values.permute(0, 2, 1, 3)
        omega = queries @ keys.transpose( 2, 3)
        # mask
        omega.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        # scaled att weight
        att_weight = self.dropout(torch.softmax(omega / keys.shape[-1] ** 0.5, dim=-1))
        # (B, no_of_heads, T, head_dim) -> (B, T, no_of_heads, head_dim)
        out = (att_weight @ values).transpose(1, 2)
        return self.out_proj(out.contiguous().view(B, T, self.d_out))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(Tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.epsilon = 1e-5
        self.shift = nn.Parameter(torch.zeros(embed_dim))
        self.scale = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.epsilon)
        return self.scale * norm + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.mhma = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_len=cfg.context_length,
            dropout=cfg.drop_rate,
            n_head=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg.emb_dim)
        self.layer_norm1 = LayerNorm(embed_dim=cfg.emb_dim)
        self.layer_norm2 = LayerNorm(embed_dim=cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mhma(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        return x + shortcut


class FeedForward(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.linear1 = nn.Linear(n_dim, 4 * n_dim)
        self.gelu = GeLU()
        self.linear2 = nn.Linear(4 * n_dim, n_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embed = nn.Embedding(config.context_length, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = LayerNorm(config.emb_dim)
        self.linear = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x: Tensor):
        B, T = x.shape
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(torch.arange(T, device=x.device))
        x = tok_embed + pos_embed
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.linear(x)
        return logits
