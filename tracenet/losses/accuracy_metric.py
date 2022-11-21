import torch

from ._utils import get_src_permutation_idx
from .matcher import HungarianMatcher


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
        return src_traces[keep], target_traces[keep]

    @torch.no_grad()
    def compute_cardinality_error(self, outputs, targets):
        device = outputs['pred_logits'].device
        tgt_lengths = torch.as_tensor([v.shape[0] for v in targets["trace"]], device=device)
        probas = outputs['pred_logits'].softmax(-1)[:, :, 1:]
        keep = probas > self.min_prob
        src_lengths = torch.as_tensor([pr[k].shape[0] for pr, k in zip(probas, keep)], device=device)
        card_err = torch.abs(src_lengths - tgt_lengths)
        self.append('cardinality error', card_err)
        self.append('relative cardinality error', card_err.float() / tgt_lengths)

    def __call__(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        # num_boxes = get_num_boxes(targets, outputs['pred_traces'].device)
        # src_traces, target_traces = self.get_matching_traces(outputs, targets, indices)
        self.compute_cardinality_error(outputs, targets)
