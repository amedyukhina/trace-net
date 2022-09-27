import torch
import torch.nn.functional as F


def symmetric_distance(source, target):
    sources = torch.stack([source, source.roll(2, -1)])
    with torch.no_grad():
        ls = F.mse_loss(sources, torch.stack([target, target]), reduction='none').sum(-1)
    ind = ls.argmin(0)
    return torch.sqrt(((sources[ind, torch.arange(len(ind))] -
                        target) ** 2).reshape(-1, 2).sum(-1)).reshape(-1, 2).sum(-1)
