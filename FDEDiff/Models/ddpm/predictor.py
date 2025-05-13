import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ddpm.model_utils import FixedTimeEmbeddings


class MLP_noise(nn.Module):
    def __init__(
        self,
        seq_len, 
        input_dim, # (B, seq_len, input_dim)
        hidden_dim = 128,
        timesteps = 1000,
        dropout = 0.1
    ):
        super(MLP_noise, self).__init__()
        self.embeddings = FixedTimeEmbeddings(hidden_dim, timesteps)
        self.conv = nn.Conv1d(
            input_dim, 
            input_dim,
            kernel_size = 3,
            padding = 1
        )
        self.dropout = nn.Dropout(p=dropout)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * seq_len)
        )
    
    def forward(self, xt, t):
        B, L, D = xt.shape
        emb = self.embeddings(t) # (B, hidden)
        xt = xt.permute(0, 2, 1) # (B, input_dim, L)
        xt = self.conv(xt).permute(0, 2, 1)
        xt = self.dropout(xt)
        xt = xt.reshape(B, -1) # (B, L * input_dim)
        xt = self.mlp1(xt) # (B, hidden_dim)
        xt = xt + emb # add time embedding
        xt = self.mlp2(xt) # (B, L * input_dim)
        return xt.view(B, L, D)