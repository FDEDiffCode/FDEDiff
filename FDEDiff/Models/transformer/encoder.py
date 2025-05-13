import torch
import torch.nn as nn
import einops

from Models.transformer.embed import PatchTST_dup, DataEmbedding
from Models.transformer.attention import AttentionLayer, FullAttention
from Models.ddpm.model_utils import StepFuseLayer


class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        output_dim,
        d_model,
        input_dim,
        num_layer,
        patch_len = 8,
        dropout = 0.1,
        n_heads = 6
    ):
        super(TimeSeriesDecoder, self).__init__()
        self.patch_len = patch_len
        self.input_layer = DataEmbedding(output_dim, d_model)
        self.encoders = []
        self.num_layer = num_layer
        for i in range(num_layer):
            self.encoders.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag = False
                        ),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    4 * d_model,
                    dropout
                )
            )
        self.encoders = nn.ModuleList(self.encoders)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, patch_len * input_dim)

    def forward(self, x, t = None): # (B, L / patch_len, output_dim)
        x = self.input_layer(x) # (B, L / patch_len, d_model)
        for enc in self.encoders:
            x = enc(x)
        x = self.norm(x) # (B, L / patch_len, d_model)
        x = self.decoder(x) # (B, L / patch_len, input_dim * patch_len)
        x = einops.rearrange(x, 'B L_p (i p) -> B (L_p p) i', p = self.patch_len) # (B, L, input_dim)
        return x

class TimeSeriesEncoderWithCond(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_layer,
        patch_len,
        n_heads,
        dropout = 0.1,
    ):
        super(TimeSeriesEncoderWithCond, self).__init__()
        self.patch = PatchTST_dup(patch_len, patch_len)
        self.input_layer = DataEmbedding(input_dim * patch_len, d_model, dropout)
        self.patch2 = PatchTST_dup(patch_len, patch_len)
        self.input_layer2 = DataEmbedding(output_dim, d_model) # 对LF进行Patch,Embedding
        self.encoders = []
        self.resblocks = [] # add step information
        self.num_layer = num_layer
        for i in range(num_layer):
            self.resblocks.append(
                StepFuseLayer(d_model) 
            )
            self.encoders.append(
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False
                        ),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False
                        ),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    4 * d_model,
                    dropout
                )
            )
        self.resblocks = nn.ModuleList(self.resblocks)
        self.encoders = nn.ModuleList(self.encoders)
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x, t, cond): # (B, L, input_dim)
        x = self.patch(x) # (B, L / patch_len, input_dim * patch_len)
        x = self.input_layer(x) # (B, L / patch_len, d_model)
        cond = self.patch2(cond)
        cond = self.input_layer2(cond)
        for i, enc in enumerate(self.encoders):
            x = self.resblocks[i](x, t)
            x = enc(x, cond)
        x = self.fc(x)
        return x # (B, L / patch_len, output_dim)

class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        output_dim,
        num_layer,
        patch_len,
        n_heads,
        dropout = 0.1,
    ):
        super(TimeSeriesEncoder, self).__init__()
        self.patch = PatchTST_dup(patch_len, patch_len)
        self.input_layer = DataEmbedding(input_dim * patch_len, d_model, dropout)
        self.encoders = []
        self.resblocks = [] # add step information
        self.num_layer = num_layer
        for i in range(num_layer):
            self.resblocks.append(
                StepFuseLayer(d_model) 
            )
            self.encoders.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False
                        ),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    4 * d_model,
                    dropout
                )
            )
        self.resblocks = nn.ModuleList(self.resblocks)
        self.encoders = nn.ModuleList(self.encoders)
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x, t = None): # (B, L, input_dim)
        x = self.patch(x) # (B, L / patch_len, input_dim * patch_len)
        x = self.input_layer(x) # (B, L / patch_len, d_model)
        for i, enc in enumerate(self.encoders):
            if t is None:
                x = enc(x)
            else:
                x = self.resblocks[i](x, t)
                x = enc(x)
        x = self.fc(x)
        return x # (B, L / patch_len, output_dim)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_hidden,
        dropout,
    ):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x): # (B, L, d_model)
        _x = x
        x = self.attention(x, x, x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        x = self.norm2(x + _x)

        return x # (B, L, d_model)
    
class DecoderLayer(nn.Module):
    def __init__(
        self,
        attention1,
        attention2,
        d_model,
        d_hidden,
        dropout,
    ):
        super(DecoderLayer, self).__init__()
        self.attention0 = attention1
        self.attention1 = attention2
        self.dropout0 = nn.Dropout(p=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, cond): # (B, L, d_model)
        _x = x
        x = self.attention0(x, x, x)
        x = self.dropout0(x)
        x = self.norm0(x + _x)

        _x = x
        x = self.attention1(x, cond, cond) # q, k, v 让cond在value上做attention
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        x = self.norm2(x + _x)

        return x # (B, L, d_model)