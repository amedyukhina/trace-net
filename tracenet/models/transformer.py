import torch
from torch import nn

from .blocks.mlp import MLP
from .tracenet import PositionEmbeddingSine


class Transformer(nn.Module):

    def __init__(self, hidden_dim, n_points=2, num_queries=100, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.input_embed = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(4, 4)),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 512, kernel_size=(2, 2), stride=(2, 2)),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, hidden_dim, (1, 1)),

        )

        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.trace_embed = MLP(hidden_dim, hidden_dim, n_points * 2, 3)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 2, rounding_mode='trunc'),
                                                        normalize=True)

    def forward(self, x):
        x = self.input_embed(x)
        pos = self.position_encodings(x)
        tr_out = self.transformer((pos + x).flatten(-2, -1).transpose(-1, -2),
                                  self.query_pos.unsqueeze(0).repeat(x.shape[0], 1, 1))
        traces = self.trace_embed(tr_out).sigmoid()
        return traces
