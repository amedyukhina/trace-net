"""
Adapted from https://github.com/facebookresearch/detr/blob/8a144f83a287f4d3fece4acdf073f387c5af387d/models/detr.py#L83
"""
import torch
import torch.nn.functional as F
from torch import nn

from ._utils import get_num_boxes, get_matching_traces, get_src_permutation_idx
from .matcher import HungarianMatcher
from .symmetric_distance import symmetric_distance
from ..utils.points import (
    get_first_and_last,
    trace_distance,
    trace_distance_param,
    point_spacing_std,
    line_straightness_mh,
    bezier_curve_from_control_points
)


class Criterion(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) compute loss between each matched pair
    """

    def __init__(self, num_classes, matcher=None, losses=None, eos_coef=0.1,
                 symmetric=False, bezier=False, lim_strt=5):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            lim_strt: limit for the straightness loss. The loss will be applied only to values larger than this number.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.eos_coef = eos_coef
        self.losses = list(losses) + ['cardinality'] if losses is not None \
            else ['loss_class', 'loss_trace_distance', 'loss_point_spacing', 'loss_end_coords', 'cardinality']
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.symmetric = symmetric
        self.bezier = bezier
        self.lim_strt = lim_strt

    def loss_class(self, outputs, targets, indices, **_):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["trace_class"], indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'loss_class': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, **_):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v) for v in targets["trace_class"]], device=device)

        # Count the number of predictions that are NOT "no-object"
        card_pred = (pred_logits.argmax(-1) > 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_trace_distance(self, outputs, targets, indices, num_boxes, **_):
        """Compute the losses related to the trace coordinates.
           Targets dicts must contain the key "trace" containing a tensor of dim [nb_target_traces, n_points * 2]
           The target traces are expected in format (y1, x1, y2, x2 ... yn, xn), normalized by the image size.
        """
        src_traces, target_traces = get_matching_traces(outputs, targets, indices)
        if self.bezier:
            assert src_traces.shape[1] == 8
            b = bezier_curve_from_control_points(src_traces.reshape(-1, 4, 2), 20)
            if self.symmetric:
                srctr = get_first_and_last(src_traces.detach())
                tgtr = get_first_and_last(target_traces.detach())
                tgtr = torch.stack([tgtr, tgtr.roll(2, -1)])
                dist = torch.abs(srctr - tgtr).sum(-1).to(src_traces.device)
                ind = torch.stack([dist.argmin(0),
                                   torch.arange(dist.shape[1]).to(src_traces.device)]).to(src_traces.device)
                target_traces = target_traces.reshape(target_traces.shape[0], -1, 2)
                target_traces = torch.stack([target_traces, target_traces.flip(1)]).to(src_traces.device)
                target_traces = target_traces[tuple(ind)]
            else:
                target_traces = target_traces.reshape(target_traces.shape[0], -1, 2)
            loss_trace = trace_distance_param(b, target_traces)
        else:
            loss_trace = trace_distance(src_traces, target_traces)

        losses = {'loss_trace_distance': loss_trace.sum() / num_boxes}

        return losses

    def loss_point_spacing(self, outputs, targets, indices, num_boxes, **_):
        src_traces, _ = get_matching_traces(outputs, targets, indices)
        loss_point_spacing = point_spacing_std(src_traces)
        losses = {'loss_point_spacing': loss_point_spacing.sum() / num_boxes}

        return losses

    def loss_straightness(self, outputs, targets, indices, num_boxes, **_):
        src_traces, _ = get_matching_traces(outputs, targets, indices)

        if self.bezier:
            assert src_traces.shape[1] == 8
            src_traces = bezier_curve_from_control_points(src_traces.reshape(-1, 4, 2), 20).reshape(-1, 40)

        loss_str = line_straightness_mh(src_traces)
        losses = {'loss_straightness': loss_str.sum() / num_boxes}

        return losses

    def loss_end_coords(self, outputs, targets, indices, num_boxes, **_):
        """Compute the end coordinate distance.
        """
        src_traces, target_traces = get_matching_traces(outputs, targets, indices)
        if self.symmetric:
            loss_trace = symmetric_distance(get_first_and_last(src_traces), get_first_and_last(target_traces))
        else:
            loss_trace = F.l1_loss(get_first_and_last(src_traces), get_first_and_last(target_traces),
                                   reduction='none')
        losses = {'loss_end_coords': loss_trace.sum() / num_boxes / 4}  # divide by 4: 2 points x 2 dimensions

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'loss_class': self.loss_class,
            'cardinality': self.loss_cardinality,
            'loss_trace_distance': self.loss_trace_distance,
            'loss_end_coords': self.loss_end_coords,
            'loss_point_spacing': self.loss_point_spacing,
            'loss_straightness': self.loss_straightness
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices=indices, num_boxes=num_boxes, **kwargs)

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        num_boxes = get_num_boxes(targets, outputs['pred_traces'].device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses
