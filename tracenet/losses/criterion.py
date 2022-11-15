"""
Adapted from https://github.com/facebookresearch/detr/blob/8a144f83a287f4d3fece4acdf073f387c5af387d/models/detr.py#L83
"""
import torch
import torch.nn.functional as F
from torch import nn

from .matcher import HungarianMatcher
from .symmetric_distance import symmetric_distance
from ..utils.points import get_first_and_last, point_segment_dist, point_spacing_std


class Criterion(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) compute loss between each matched pair
    """

    def __init__(self, num_classes, matcher=None, losses=None, eos_coef=0.1, symmetric=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
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

    def loss_class(self, outputs, targets, indices, **_):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
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
        src_traces, target_traces = self._get_matching_traces(outputs, targets, indices)
        x = torch.stack([target_traces[:, 2 * i:2 * i + 4]
                         for i in range(int(target_traces.shape[-1] / 2) - 1)])  # get all segments of the target
        v = x[:, :, :2]  # first points of all target segments
        w = x[:, :, 2:]  # second points of all target segments
        points = [src_traces[:, 2 * j:2 * j + 2]
                  for j in range(int(src_traces.shape[-1] / 2))]  # all points of the source trace
        loss_trace = torch.stack([
            torch.stack([point_segment_dist(vv, ww, p)
                         for vv, ww in zip(v, w)]).min(0)[0]  # dist from each point to the closest segment
            for p in points])  # calculate for all points

        if (loss_trace == 0).any():
            print('Loss Trace = 0:', loss_trace)
            with torch.no_grad():
                print(loss_trace.min(0))
                print(loss_trace.min(1))
                print(v)
                print(w)
                print(v == w)
        loss_trace = loss_trace.mean(0)  # average among all points
        losses = {'loss_trace_distance': loss_trace.sum() / num_boxes}

        return losses

    def loss_point_spacing(self, outputs, targets, indices, num_boxes, **_):
        src_traces, _ = self._get_matching_traces(outputs, targets, indices)
        loss_point_spacing = point_spacing_std(src_traces)
        losses = {'loss_point_spacing': loss_point_spacing.sum() / num_boxes}

        return losses

    def loss_end_coords(self, outputs, targets, indices, num_boxes, **_):
        """Compute the end coordinate distance.
        """
        src_traces, target_traces = self._get_matching_traces(outputs, targets, indices)
        if self.symmetric:
            loss_trace = symmetric_distance(get_first_and_last(src_traces), get_first_and_last(target_traces))
        else:
            loss_trace = F.mse_loss(get_first_and_last(src_traces), get_first_and_last(target_traces),
                                    reduction='none')
        losses = {'loss_end_coords': loss_trace.sum() / num_boxes}

        return losses

    def _get_matching_traces(self, outputs, targets, indices):
        assert 'pred_traces' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_traces = outputs['pred_traces'][idx]
        target_traces = torch.cat([t[i] for t, (_, i) in zip(targets['trace'], indices)], dim=0)
        return src_traces, target_traces

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'loss_class': self.loss_class,
            'cardinality': self.loss_cardinality,
            'loss_trace_distance': self.loss_trace_distance,
            'loss_end_coords': self.loss_end_coords,
            'loss_point_spacing': self.loss_point_spacing
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices=indices, num_boxes=num_boxes, **kwargs)

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(targets['trace'][i].shape[0] for i in range(len(targets['trace'])))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses
