import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embedding,
        embedding_dim,
        commitment_cost
    ):
        super(VectorQuantizer, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embedding, 
            1.0 / num_embedding
        )
        self.commitment_cost = commitment_cost

    def forward(self, x): # (B, L, embedding_dim)
        B, L, D = x.shape
        flat_x = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flat_x, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1).view(B, L)
        zq = self.embedding(encoding_indices) # (B, L, embedding_dim)

        decoder_input = x + (zq - x).detach() # (B, L, embedding_dim)
        e_loss = F.mse_loss(zq.detach(), x)
        q_loss = F.mse_loss(zq, x.detach())
        loss = self.commitment_cost * e_loss + q_loss

        return decoder_input, loss