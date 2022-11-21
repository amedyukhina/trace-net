import torch


def get_num_boxes(targets, device):
    # Compute the average number of target boxes across all nodes, for normalization purposes
    num_boxes = sum(targets['trace'][i].shape[0] for i in range(len(targets['trace'])))
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
    num_boxes = torch.clamp(num_boxes, min=1).item()
    return num_boxes


def get_matching_traces(outputs, targets, indices):
    assert 'pred_traces' in outputs
    idx = get_src_permutation_idx(indices)
    src_traces = outputs['pred_traces'][idx]
    target_traces = torch.cat([t[i] for t, (_, i) in zip(targets['trace'], indices)], dim=0)
    return src_traces, target_traces


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
