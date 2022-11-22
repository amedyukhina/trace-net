import torch

from ._utils import get_src_permutation_idx
from .matcher import HungarianMatcher
from .symmetric_distance import symmetric_distance
from ..utils.points import get_first_and_last


class Metric:

    def __init__(self, matcher=None, min_prob=0.7):
        self.buffer = dict()
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.min_prob = min_prob
        self.mean = dict()
        self.std = dict()

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
    def get_src_and_target_lengths(self, outputs, targets):
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

    @torch.no_grad()
    def compute_end_distance(self, src_traces, target_traces, batch_idx):
        end_dist = symmetric_distance(get_first_and_last(src_traces), get_first_and_last(target_traces))
        end_dist = torch.as_tensor([end_dist[batch_idx == i].mean() for i in batch_idx.unique()])
        self.append('end distance', end_dist)

    def __call__(self, outputs, targets):
        src_lengths, tgt_lengths = self.get_src_and_target_lengths(outputs, targets)
        self.compute_cardinality_error(src_lengths, tgt_lengths)
        indices = self.matcher(outputs, targets)
        src_traces, target_traces, batch_idx = self.get_matching_traces(outputs, targets, indices)
        self.compute_pr(src_lengths, tgt_lengths, batch_idx)
        self.compute_end_distance(src_traces, target_traces, batch_idx)
