from functools import reduce

import torch
from torch import nn

from .blocks.mlp import MLP
from .blocks.pos_encodings import PositionEmbeddingSine


class TraceNet(nn.Module):
    """
    TraceNet class
    """

    def __init__(self, backbone, transformer_input_layer,
                 hidden_dim, num_classes, n_points=2, num_queries=100, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, pos_coef=0.1):
        super().__init__()

        self.backbone = backbone
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.transformer_inputs = []
        get_module_by_name(backbone, transformer_input_layer).register_forward_hook(self._get_transformer_inputs)

        # prediction heads, one extra class for predicting non-empty slots
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.trace_embed = MLP(hidden_dim, hidden_dim, n_points * 2, 3)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 2, rounding_mode='trunc'),
                                                        normalize=True)

        self.pos_coef = pos_coef

    def _get_transformer_inputs(self, module, inp, output):
        self.transformer_inputs.append(torch.stack([m for m in output]))

    def forward(self, x):
        bb_out = self.backbone(x)
        tr_input = self.transformer_inputs[-1]
        self.transformer_inputs = []
        pos = self.position_encodings(tr_input)
        tr_out = self.transformer((pos + self.pos_coef * tr_input).flatten(-2, -1).transpose(-1, -2),
                                  self.query_pos.unsqueeze(0).repeat(tr_input.shape[0], 1, 1))
        traces = self.trace_embed(tr_out).sigmoid()
        class_prob = self.class_embed(tr_out)
        return {'pred_logits': class_prob, 'pred_traces': traces, 'backbone_out': bb_out}


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
