import math

import torch
from torch import nn

from .blocks.mlp import MLP
from .tracenet import PositionEmbeddingSine
from ..datasets.transforms import reshape_image_for_transformer


class Transformer(nn.Module):

    def __init__(self, hidden_dim, imgsize=16, n_points=2, num_queries=100, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, pos_coef=0.1):
        super().__init__()

        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.trace_embed = MLP(hidden_dim, hidden_dim, n_points * 2, 3)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 2, rounding_mode='trunc'),
                                                        normalize=True)

        self.pos_coef = pos_coef
        self.imgsize = imgsize
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if not x.shape[-1] == self.imgsize * math.sqrt(self.hidden_dim):
            raise ValueError(rf"Input image shape must be {self.imgsize * math.sqrt(self.hidden_dim)}")
        x = reshape_image_for_transformer(x, self.imgsize)
        pos = self.position_encodings(x)
        tr_out = self.transformer((pos + self.pos_coef * x).flatten(-2, -1).transpose(-1, -2),
                                  self.query_pos.unsqueeze(0).repeat(x.shape[0], 1, 1))
        traces = self.trace_embed(tr_out).sigmoid()
        return traces
