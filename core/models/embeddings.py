import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, max_norm=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.max_norm = max_norm

        self.emb_scale = d_proj ** 0.5
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        self.emb_layers.append(
            nn.Embedding(n_token, d_embed, padding_idx=0)
        )
        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))

    def forward(self, inp):
        embed = self.emb_layers[0](inp)
        # L2 normalize user embedding
        # if self.max_norm:
        #     embed = F.normalize(embed)
        if self.d_proj != self.d_embed:
            embed  = F.linear(embed, self.emb_projs[0])

        embed.mul_(self.emb_scale)

        return embed

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]
        return output
