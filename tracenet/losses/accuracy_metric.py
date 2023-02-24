import torch
import torch.nn.functional as F

from ._utils import get_src_permutation_idx
from .matcher import HungarianMatcher
from ..utils.points import get_first_and_last, trace_distance, bezier_curve_from_control_points, trace_distance_param


class Metric:
    '''
    Calculates various accuracy metrics for predicted vs ground truth traces.
    '''

    def __init__(self, matcher=None, bezier=False, min_prob=0.7, dist_scaling=1.):
        self.buffer = dict()
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.min_prob = min_prob
        self.dist_scaling = dist_scaling
        self.mean = dict()
        self.std = dict()
        self.bezier = bezier

    def reset(self):
        self.buffer = dict()

    def append(self, key, values):
        if key in self.buffer:
            self.buffer[key] = torch.concat([self.buffer[key], values])
        else:
            self.buffer[key] = values

    @torch.no_grad()
    def aggregate(self):
        for key in self.buffer.keys():
            self.mean[key] = torch.mean(self.buffer[key].float()).item()
            self.std[key] = torch.std(self.buffer[key].float()).item()

    @torch.no_grad()
    def get_matching_traces(self, outputs, targets, indices):
        idx = get_src_permutation_idx(indices)
        src_traces = outputs['pred_traces'][idx]
        src_probs = outputs['pred_logits'][idx].softmax(-1)[:, 1:]
        keep = src_probs.max(-1).values > self.min_prob
        target_traces = torch.cat([t[i] for t, (_, i) in zip(targets['trace'], indices)], dim=0)
        return src_traces[keep], target_traces[keep], idx[0][keep]

    @torch.no_grad()
    def get_src_and_target_numbers(self, outputs, targets):
        device = outputs['pred_logits'].device
        tgt_lengths = torch.as_tensor([v.shape[0] for v in targets["trace"]], device=device)
        probas = outputs['pred_logits'].softmax(-1)[:, :, 1:]
        keep = probas > self.min_prob
        src_lengths = torch.as_tensor([pr[k].shape[0] for pr, k in zip(probas, keep)], device=device)
        return src_lengths.float(), tgt_lengths.float()

    @torch.no_grad()
    def compute_cardinality_error(self, src_lengths, tgt_lengths):
        card_err = torch.abs(src_lengths - tgt_lengths)
        self.append('cardinality error', card_err)
        self.append('relative cardinality error', card_err.float() / tgt_lengths)
        self.append('number of filaments GT', tgt_lengths)
        self.append('number of detected filaments', src_lengths)

    @torch.no_grad()
    def compute_pr(self, src_lengths, tgt_lengths, batch_idx):
        matched = torch.as_tensor([(batch_idx == i).sum() for i in range(src_lengths.shape[0])],
                                  device=tgt_lengths.device)
        recall = matched / tgt_lengths
        precision = matched / src_lengths
        f1score = 2 * precision * recall / (precision + recall)
        self.append('Precision', precision)
        self.append('Recall', recall)
        self.append('F1 Score', f1score)
        self.append('number of matched filaments', matched)

    @torch.no_grad()
    def compute_end_distance(self, src_traces, target_traces):
        end_dist = symmetric_distance(get_first_and_last(src_traces), get_first_and_last(target_traces))
        self.append('end error', end_dist * self.dist_scaling)

    @torch.no_grad()
    def compute_trace_distance(self, src_traces, target_traces):
        if self.bezier:
            trace_dist = trace_distance_param(src_traces.reshape(src_traces.shape[0], -1, 2),
                                              target_traces.reshape(target_traces.shape[0], -1, 2))
        else:
            trace_dist = trace_distance(src_traces, target_traces)
        self.append('trace distance error', trace_dist * self.dist_scaling)

    @torch.no_grad()
    def compute_length(self, src_traces, target_traces):
        src_len = curve_length(src_traces)
        tgt_len = curve_length(target_traces)
        src_end_dist = dist_ends(src_traces)
        tgt_end_dist = dist_ends(target_traces)
        src_curvature = src_len / src_end_dist
        tgt_curvature = tgt_len / tgt_end_dist
        self.append('filament length GT', tgt_len * self.dist_scaling)
        self.append('filament length detected', src_len * self.dist_scaling)
        self.append('end length GT', tgt_end_dist * self.dist_scaling)
        self.append('end length detected', src_end_dist * self.dist_scaling)
        self.append('filament length error', torch.abs(src_len - tgt_len) * self.dist_scaling)
        self.append('relative filament length error', torch.abs(src_len - tgt_len) / tgt_len)
        self.append('filament end error', torch.abs(src_end_dist - tgt_end_dist) * self.dist_scaling)
        self.append('relative filament end error', torch.abs(src_end_dist - tgt_end_dist) / tgt_end_dist)
        self.append('curvature GT', tgt_curvature)
        self.append('curvature detected', src_curvature)
        self.append('curvature error', torch.abs(src_curvature - tgt_curvature))

    def __call__(self, outputs, targets):
        src_lengths, tgt_lengths = self.get_src_and_target_numbers(outputs, targets)
        self.compute_cardinality_error(src_lengths, tgt_lengths)
        indices = self.matcher(outputs, targets)
        src_traces, target_traces, batch_idx = self.get_matching_traces(outputs, targets, indices)
        if self.bezier:
            assert src_traces.shape[1] == 8
            src_traces = bezier_curve_from_control_points(src_traces.reshape(-1, 4, 2), 20).reshape(-1, 40)
        self.compute_pr(src_lengths, tgt_lengths, batch_idx)
        self.compute_end_distance(src_traces, target_traces)
        self.compute_trace_distance(src_traces, target_traces)
        self.compute_length(src_traces, target_traces)


def symmetric_distance(source, target):
    npoints = int(source.shape[-1] / 2)
    source = source.reshape(-1, npoints, 2)
    sources = torch.stack([source, torch.flip(source, [1])]).reshape(2, -1, npoints * 2)
    with torch.no_grad():
        ls = F.mse_loss(sources, torch.stack([target, target]), reduction='none').sum(-1)
    ind = ls.argmin(0)
    return torch.sqrt(((sources[ind, torch.arange(len(ind))] -
                        target) ** 2).reshape(-1, 2).sum(-1)).reshape(-1, npoints).mean(-1)


def curve_length(x):
    n = int(x.shape[-1] / 2)
    return torch.sqrt(((x[:, 2:] - x[:, :-2]) ** 2).reshape(-1, 2).sum(-1)).reshape(-1, n - 1).sum(-1)


def dist_ends(x):
    return torch.sqrt(((x[:, :2] - x[:, -2:]) ** 2).sum(-1))
