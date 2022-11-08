import torch
from torch import nn

from tracenet.models.blocks.mlp import MLP


class DETR(nn.Module):

    def __init__(self, n_points=2, n_classes=1):
        super().__init__()
        self.detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        hdim = self.detr.transformer.d_model
        self.detr.class_embed = torch.nn.Linear(hdim, n_classes + 1)
        self.detr.bbox_embed = MLP(hdim, hdim, n_points * 2, 3)

    def forward(self, x):
        out = self.detr(x)
        return {'pred_logits': out['pred_logits'], 'pred_traces': out['pred_boxes']}
