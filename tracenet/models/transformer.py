import torch
from torch import nn

from .blocks.detr_transformer import Transformer as DETR_Transformer
from .blocks.mlp import MLP
from .blocks.pos_encodings import PositionEmbeddingSine


class Transformer(nn.Module):

    def __init__(self, hidden_dim, n_points=2, num_queries=10, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # self.input_embed = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(4, 4)),
        #     nn.PReLU(),
        #     nn.Dropout(0.2),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16, 64, kernel_size=(4, 4), stride=(4, 4)),
        #     nn.PReLU(),
        #     nn.Dropout(0.2),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 512, kernel_size=(2, 2), stride=(2, 2)),
        #     nn.PReLU(),
        #     nn.Dropout(0.2),
        #     nn.BatchNorm2d(512),
        #     nn.Conv2d(512, hidden_dim, (1, 1)),
        #
        # )
        self.input_embed = nn.Conv2d(1, hidden_dim, kernel_size=(1, 1))

        self.transformer = DETR_Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.trace_embed = MLP(hidden_dim, hidden_dim, n_points * 2, 3)

        # output positional encodings (object queries)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 2, rounding_mode='trunc'),
                                                        normalize=True)

    def forward(self, x):
        x = self.input_embed(x)
        pos = self.position_encodings(x)
        tr_out = self.transformer(x, mask=None, query_embed=self.query_embed.weight, pos_embed=pos)[0][-1]
        traces = self.trace_embed(tr_out).sigmoid()
        return traces
