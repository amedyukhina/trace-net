import torch
import torch.nn.functional as F


def symmetric_distance(source, target):
    npoints = int(source.shape[-1] / 2)
    source = source.reshape(-1, npoints, 2)
    sources = torch.stack([source, torch.flip(source, [1])]).reshape(2, -1, npoints * 2)
    with torch.no_grad():
        ls = F.l1_loss(sources, torch.stack([target, target]), reduction='none').sum(-1)
    ind = ls.argmin(0)
    return (torch.abs(sources[ind, torch.arange(len(ind))]
                      - target).reshape(-1, 2).sum(-1)).reshape(-1, npoints).mean(-1)
