import torch
import numpy as np
from tqdm import trange


def replace_lora(model: torch.nn.Module, LinearLora, rank, alpha):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.Linear):
            setattr(model, name, LinearLora(layer, rank, alpha))
        else:
            replace_lora(layer, LinearLora, rank, alpha)


def generate(
    idx: torch.Tensor, model, context_length, max_output_token, temperate=0.1, top_k=5
):
    assert top_k >= 1
    if not temperate:
        for _ in trange(max_output_token):
            idx_inp = idx[:, -context_length:]
            logits = model(idx_inp)
            # (batch, T, out_dim) -> (batch, out_dim) of final element in seq
            logits = logits[:, -1, :]
            proba = torch.softmax(logits, dim=-1)
            out_tok = torch.argmax(proba, dim=-1, keepdim=True)
            idx = torch.concat((idx, out_tok), dim=1)
        return idx
    else:
        for _ in trange(max_output_token):
            idx_inp = idx[:, -context_length:]
            logits = model(idx_inp)
            # (batch, T, out_dim) -> (batch, out_dim) of final element in seq
            logits = logits[:, -1, :]
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
            proba = torch.softmax(logits, dim=-1)
            out_tok = torch.multinomial(proba, num_samples=1)
            idx = torch.concat((idx, out_tok), dim=1)
        return idx


def generate_new_text(idx: torch.Tensor, model, context_length, max_output_token):
    for _ in trange(max_output_token):
        idx_inp = idx[:, -context_length:]
        logits = model(idx_inp)
        # (batch, T, out_dim) -> (batch, out_dim) of final element in seq
        logits = logits[:, -1, :]
        # proba = torch.softmax(logits, dim=-1)
        out_tok = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.concat((idx, out_tok), dim=1)
    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, " "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params["wpe"])
    gpt.tok_embed.weight = assign(gpt.tok_embed.weight, params["wte"])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mhma.W_queries.weight = assign(
            gpt.transformer_blocks[b].mhma.W_queries.weight, q_w.T
        )
        gpt.transformer_blocks[b].mhma.W_keys.weight = assign(
            gpt.transformer_blocks[b].mhma.W_keys.weight, k_w.T
        )
        gpt.transformer_blocks[b].mhma.W_values.weight = assign(
            gpt.transformer_blocks[b].mhma.W_values.weight, v_w.T
        )
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].mhma.W_queries.bias = assign(
            gpt.transformer_blocks[b].mhma.W_queries.bias, q_b
        )
        gpt.transformer_blocks[b].mhma.W_keys.bias = assign(
            gpt.transformer_blocks[b].mhma.W_keys.bias, k_b
        )
        gpt.transformer_blocks[b].mhma.W_values.bias = assign(
            gpt.transformer_blocks[b].mhma.W_values.bias, v_b
        )
        gpt.transformer_blocks[b].mhma.out_proj.weight = assign(
            gpt.transformer_blocks[b].mhma.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].mhma.out_proj.bias = assign(
            gpt.transformer_blocks[b].mhma.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )
        gpt.transformer_blocks[b].ff.linear1.weight = assign(
            gpt.transformer_blocks[b].ff.linear1.weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.linear1.bias = assign(
            gpt.transformer_blocks[b].ff.linear1.bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.transformer_blocks[b].ff.linear2.weight = assign(
            gpt.transformer_blocks[b].ff.linear2.weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.linear2.bias = assign(
            gpt.transformer_blocks[b].ff.linear2.bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )
        gpt.transformer_blocks[b].layer_norm1.scale = assign(
            gpt.transformer_blocks[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )
        gpt.transformer_blocks[b].layer_norm1.shift = assign(
            gpt.transformer_blocks[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )
        gpt.transformer_blocks[b].layer_norm2.scale = assign(
            gpt.transformer_blocks[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"],
        )
        gpt.transformer_blocks[b].layer_norm2.shift = assign(
            gpt.transformer_blocks[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"],
        )
    gpt.norm.scale = assign(gpt.norm.scale, params["g"])
    gpt.norm.shift = assign(gpt.norm.shift, params["b"])
    gpt.linear.weight = assign(gpt.linear.weight, params["wte"])


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded = torch.tensor(encoded).unsqueeze(0)  # adding batch dimension
    return encoded

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())  # removing the batch dimension