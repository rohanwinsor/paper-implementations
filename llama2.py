import torch
import torch.nn as nn
from dataclasses import dataclass

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

