import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class FixedTimeEmbeddings(nn.Module):
    def __init__(
        self,
        d_model,
        max_len = 5000,
    ):
        super(FixedTimeEmbeddings, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        embeds = self.pe[t]
        return embeds # (B, d_model)


class AdaLayerNorm_t(nn.Module):
    ''' 
        from Diffusion-TS github.
        fuse step and x.
    '''
    def __init__(self, n_embd):
        super(AdaLayerNorm_t, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd * 2)
        )
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, t_emb): # (B, L, D) and (B, D)
        emb = self.transform(t_emb).unsqueeze(1) # (B, 1, 2 * D)
        scale, shift = torch.chunk(emb, 2, dim=2) # (B, 1, D), (B, 1, D)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class StepFuseLayer(nn.Module):
    '''
        embedding step
        fuse step and x
        resblock structure
    '''
    def __init__(self, d_model):
        super(StepFuseLayer, self).__init__()
        self.step_embeddings = FixedTimeEmbeddings(d_model)
        self.step_ln = AdaLayerNorm_t(d_model)
        self.x_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x, t): # (B, L, d_model) and (B, )
        h = self.x_layer(x)
        emb = self.step_embeddings(t)
        h = self.step_ln(h, emb)
        x = x + h
        return x # (B, L, d_model)