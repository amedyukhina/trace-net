# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy-paste from https://github.com/facebookresearch/detr/blob/main/models/detr.py
"""
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """
    Multilayer perceptron.
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
