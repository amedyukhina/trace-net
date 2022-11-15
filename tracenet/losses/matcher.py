# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.

Adapted from https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..utils.points import get_first_and_last


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_coord: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_coord = get_first_and_last(outputs["pred_traces"].flatten(0, 1))

        # Also concat the target labels and boxes
        tgt_coord = get_first_and_last(torch.cat(targets['trace']))
        tgt_ids = torch.cat(targets['trace_class'])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes

        cost_coord = torch.cdist(torch.stack([out_coord, out_coord.roll(2, -1)]),
                                 torch.stack([tgt_coord, tgt_coord]), p=1)
        cost_coord = torch.min(cost_coord, dim=0)[0]

        # Final cost matrix
        C = self.cost_coord * cost_coord + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets["trace"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
