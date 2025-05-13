import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag = False
    ):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        _, S, _, D = v.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', q, k)
        if self.mask_flag:
            mask = TriangularCausalMask(B, L, q.device)
            scores.masked_fill_(mask.mask, -np.inf)
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum('bhls,bshd->bhld', scores, v)
        return V.contiguous()


class AttentionLayer(nn.Module): # multi-head attention layer
    def __init__(
        self,
        attention,
        d_model,
        n_heads
    ):
        super(AttentionLayer, self).__init__()
        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model) # d_model = n_heads * (d_model // n_heads)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads
        q = self.q_proj(q).view(B, L, H, -1)
        k = self.k_proj(k).view(B, S, H, -1)
        v = self.v_proj(v).view(B, S, H, -1)
        out = self.attention(q, k, v).view(B, L, -1)
        return self.o_proj(out)


