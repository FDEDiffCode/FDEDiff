import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math


class PatchTST_dup(nn.Module):
    def __init__(
        self,
        patch_len=8,
        stride=8  # stride = patch_len
    ):
        super(PatchTST_dup, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):  # (B, L, input_dim), stride = patch_len | L
        x = x.unfold(1, self.patch_len, self.stride)
        B, L, D, C = x.shape
        # return x.view(B, L, -1) # (B, L // patch_len, input_dim * patch_len)
        # view报错：view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # (B, L // patch_len, input_dim * patch_len)
        return x.reshape(B, L, -1)


class PatchTST(nn.Module):
    def __init__(
        self,
        input_dim,
        patch_len=8,
        stride=8
    ):
        super(PatchTST, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=patch_len,
            stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x):  # (B, L, input_dim)
        x = x.permute(0, 2, 1)  # (B, input_dim, L)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (B, patched L, input_dim)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        max_len=5000
    ):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -
            (math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]  # (1, L, d_model)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model
    ):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        dropout=0.1
    ):
        super(DataEmbedding, self).__init__()
        self.position_embedding = PositionalEmbedding(d_model)
        self.token_embedding = TokenEmbedding(input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # (B, L, D)
        x = self.token_embedding(x)
        x = x + self.position_embedding(x)
        return self.dropout(x)  # (B, L, d_model)
