import json
import math
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

    def __init__(self, n_points=2, n_classes=1, pretrained=True, pretrained_model_path=None, bezier=False):
        super().__init__()
        self.detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=pretrained)
        hdim = self.detr.transformer.d_model
        self.detr.class_embed = torch.nn.Linear(hdim, n_classes + 1)

        if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
            with open(os.path.join(os.path.dirname(pretrained_model_path), 'config.json')) as f:
                config = json.load(f)
            self.detr.bbox_embed = MLP(hdim, hdim,
                                       config['n_points'] * 2 if not config['bezier'] else 8,
                                       3)
            print(rf"Loading pretrained backbone weights from {pretrained_model_path}")
            self.load_state_dict(torch.load(pretrained_model_path))

        self.detr.bbox_embed = MLP(hdim, hdim, n_points * 2, 3)

        theta = - math.pi / 4
        self.rot_transform = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ]).float()
        self.bezier = bezier

    def forward(self, x):
        out = self.detr(x)
        if self.bezier:
            shape = out['pred_boxes'].shape
            p0, p1, p2, p3 = torch.unbind(out['pred_boxes'].reshape(shape[0], shape[1], 4, 2), dim=-2)
            affine = _get_affine(p0, p3)
            traces = torch.concat([p0,
                                   _transform_control_point(p1, self.rot_transform, affine),
                                   _transform_control_point(p2, self.rot_transform, affine),
                                   p3], dim=-1)
        else:
            traces = out['pred_boxes']
        return {'pred_logits': out['pred_logits'], 'pred_traces': traces}


def _get_affine(p0, p3):
    p0n = p0.cpu().detach().flatten(0, 1)
    p3n = p3.cpu().detach().flatten(0, 1)
    affine = torch.stack([torch.tensor([
        [(stop[1] - start[1]) * math.sqrt(2), 0, start[1]],
        [0, (stop[0] - start[0]) * math.sqrt(2), start[0]],
        [0, 0, 1]
    ]) for start, stop in zip(p0n, p3n)])
    return affine


def _transform_control_point(p, rot_transform, affine):
    shape = p.shape
    p_n = torch.stack([(p[:, :, 1] - 0.5) * 4, p[:, :, 0], torch.ones(shape[0], shape[1]).to(p.device)], dim=-1)
    p_n = torch.matmul(rot_transform.to(p_n.device), p_n.flatten(0, 1).transpose(0, 1)).transpose(0, 1)
    p_n = torch.stack([torch.matmul(aff.to(p_n.device), p_n_i) for aff, p_n_i in zip(affine, p_n)])
    p_n = torch.flip(p_n[:, :2], dims=[-1]).reshape(shape)
    return p_n
