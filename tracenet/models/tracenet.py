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
                 num_encoder_layers=6, num_decoder_layers=6, decoder_only=False,
                 freeze_backbone=False):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        else:
            self.backbone.requires_grad_(True)

        self.decoder_only = decoder_only
        if decoder_only:
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True),
                num_layers=num_decoder_layers
            )
        else:
            self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers,
                                              num_decoder_layers, batch_first=True)
        self.transformer_inputs = []
        get_module_by_name(backbone, transformer_input_layer).register_forward_hook(self._get_transformer_inputs)

        # spatial positional encodings
        self.position_encodings = PositionEmbeddingSine(torch.div(hidden_dim, 2, rounding_mode='trunc'),
                                                        normalize=True)
        self.input_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))

        # output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)

        # prediction heads, one extra class for predicting non-empty slots
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.trace_embed = MLP(hidden_dim, hidden_dim, n_points * 2, 3)

    def _get_transformer_inputs(self, module, inp, output):
        if self.decoder_only:
            output = output[0]
        self.transformer_inputs.append(torch.stack([m for m in output]))

    def forward_decoder(self, x):
        tr_out = self.transformer(self.query_pos.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x)
        return tr_out

    def forward_all(self, x):
        pos = self.position_encodings(x)
        x = self.input_embed(x)
        tr_out = self.transformer((pos + x).flatten(-2, -1).transpose(-1, -2),
                                  self.query_pos.weight.unsqueeze(0).repeat(x.shape[0], 1, 1))
        return tr_out

    def forward(self, x):
        bb_out = self.backbone(x)
        tr_input = self.transformer_inputs[-1]
        self.transformer_inputs = []
        if self.decoder_only:
            tr_out = self.forward_decoder(tr_input)
        else:
            tr_out = self.forward_all(tr_input)
        traces = self.trace_embed(tr_out).sigmoid()
        class_prob = self.class_embed(tr_out)
        return {'pred_logits': class_prob, 'pred_traces': traces, 'backbone_out': bb_out}


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
