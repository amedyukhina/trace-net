import json
import os

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """
    Multilayer perceptron.
    Copy-paste from https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):

    def __init__(self, n_points=2, n_classes=1, pretrained_model_path=None):
        super().__init__()
        self.detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        hdim = self.detr.transformer.d_model
        self.detr.class_embed = torch.nn.Linear(hdim, n_classes + 1)

        if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
            with open(os.path.join(os.path.dirname(pretrained_model_path), 'config.json')) as f:
                config = json.load(f)
            self.detr.bbox_embed = MLP(hdim, hdim, config['n_points'] * 2, 3)
            print(rf"Loading pretrained backbone weights from {pretrained_model_path}")
            self.load_state_dict(torch.load(pretrained_model_path))

        self.detr.bbox_embed = MLP(hdim, hdim, n_points * 2, 3)

    def forward(self, x):
        out = self.detr(x)
        return {'pred_logits': out['pred_logits'], 'pred_traces': out['pred_boxes']}
