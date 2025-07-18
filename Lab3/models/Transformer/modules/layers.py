"""
Author: Chi-An Chen
Date: 2025-07-17
Description: self-designed Muti-head attention
"""

import math
import torch
import torch.nn as nn

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim 必須能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        '''
        x: (batch_size, num_tokens, dim)
        '''
        B, N, D = x.shape  # e.g., (B, N, 768)
        
        # qkv shape: (B, N, 3 * dim)
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = attn_scores.softmax(dim=-1)  # (B, num_heads, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # attention output
        out = attn_weights @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # concat heads → (B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    