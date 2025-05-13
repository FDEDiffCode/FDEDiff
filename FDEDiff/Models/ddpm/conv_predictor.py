import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ddpm.model_utils import FixedTimeEmbeddings, AdaLayerNorm_t


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        if_act = True
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            stride=2,
            kernel_size=3,
            padding=1
        )
        self.act = nn.GELU()
        self.if_act = if_act
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x): # (B, input_dim, L)
        x = self.conv(x)
        if self.if_act:
            x = self.act(x)
        x = self.norm(x)
        return x # (B, output_dim, L)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        if_act = True
    ):
        super(UpConvBlock, self).__init__()
        self.if_act = if_act
        self.norm = nn.BatchNorm1d(output_dim)
        self.act = nn.GELU()
        self.conv = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
    
    def forward(self, x): # (B, input_dim, L)
        x = self.conv(x)
        if self.if_act:
            x = self.act(x)
        x = self.norm(x)
        return x # (B, output_dim, 2 * L)


class ConvPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        d1,
        num_layer = 2,
        timesteps = 1000,
        dc = 0
    ):
        super(ConvPredictor, self).__init__()
        
        # fuse x and t
        self.embeddings = FixedTimeEmbeddings(d1, timesteps)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, d1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d1, d1, kernel_size=3, padding=1)
        )
        self.ln = AdaLayerNorm_t(d1)

        # encoder xt 
        self.conv2 = []
        last_dim = d1
        for i in range(num_layer):
            self.conv2.append(
                ConvBlock( 
                    last_dim, 
                    last_dim * 2,
                    i < num_layer - 1
                )
            )
            last_dim *= 2
        self.conv2 = nn.Sequential(*self.conv2)

        # decoder xt without cond
        last_dim = d1 * (2 ** num_layer)
        self.conv3_wo_c = []
        for i in range(num_layer):
            self.conv3_wo_c.append(
                UpConvBlock(
                    last_dim,
                    last_dim // 2,
                )
            )
            last_dim = last_dim // 2
        self.conv3_wo_c.append(nn.Conv1d(last_dim, input_dim, kernel_size=1))
        self.conv3_wo_c = nn.Sequential(*self.conv3_wo_c)
        
        # decoder xt with c
        last_dim = d1 * (2 ** num_layer)
        self.conv3_c =[]
        for i in range(num_layer):
            self.conv3_c.append(
                UpConvBlock(
                    last_dim + (dc if i == 0 else 0),
                    last_dim // 2,
                )
            )
            last_dim = last_dim // 2
        self.conv3_c.append(nn.Conv1d(last_dim, input_dim, kernel_size=1))
        self.conv3_c = nn.Sequential(*self.conv3_c)


    def forward(self, x, t, c=None): 
        '''
            x: (B, seq_len, input_dim) 
            t: (B, )
            c: (B, seq_len / (2 ** num_layer), c_dim)
        '''
        emb = self.embeddings(t) # (B, d1)
        x = x.permute(0, 2, 1) # (B, input_dim, seq_len)
        x = self.conv1(x).permute(0, 2, 1) # (B, seq_len, d1)
        xt = self.ln(x, emb) # (B, seq_len, d1) x with t
        
        # conv encoder
        xt = xt.permute(0, 2, 1)
        xt = self.conv2(xt) # (B, d1 * (2 ** num_layer), seq_len / (2 ** num_layer))
        
        # add c / conv decoder
        if c is None:
            xt = self.conv3_wo_c(xt) # (B, input_dim, seq_len)
        else:
            xt = torch.concat([xt, c], dim=1) # (B, d1 * (2 ** num_layer) + dc, seq_len / (2 ** num_layer))
            xt = self.conv3_c(xt) # (B, input_dim, seq_len)

        return xt.permute(0, 2, 1)  # (B, seq_len, input_dim)