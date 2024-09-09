import numpy as np
import torch
from torch import nn

class AGE(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, query_pos, objects_num, mask=None):

        tgt_norm = tgt / tgt.norm(dim=-1, keepdim=True)  # (64, 8, 512)

        cosine_sim_matrix = torch.matmul(tgt_norm, tgt_norm.transpose(1, 2))  # (64, 8, 10)


        inverse_similarity_matrix = cosine_sim_matrix
        inverse_similarity_matrix = torch.tanh(inverse_similarity_matrix)

        weighted_tgt = torch.bmm(inverse_similarity_matrix, tgt)
        output = weighted_tgt

        return output