import numpy as np
import torch
from torch import nn

class BiLstm(nn.Module):
    def __init__(self, d_model=512,sample_num=15,track_num=3):
        super().__init__()

        self._reset_parameters()
        self.d_model = d_model
        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=d_model // 2,num_layers=sample_num+track_num,
                              batch_first=True, bidirectional=True)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visual, query_embed, track_nums, mask=None):
        bs = visual.shape[0]  # key_visual=[B,L,D]
        query_target = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [N,B,D]
        query_target= query_target.permute(1, 0, 2)
        content_vectors = torch.cat([visual, query_target], dim=1)  # (bsz, sample_numb+object_num, hidden_dim)

        content_vectors, _ = self.bilstm(content_vectors)  # (bsz, sample_numb, hidden_dim)

        return content_vectors[:,query_embed.shape[0]:,:]