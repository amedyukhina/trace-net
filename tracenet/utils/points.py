import torch


def normalize_points(x, size):
    return x / torch.tensor(size)


def denormalize_points(x, size):
    return x * torch.tensor(size)


def get_first_and_last_points(points, labels):
    npoints = []
    for lb in labels.unique():
        point = points[labels == lb]
        if len(point) > 1:
            npoints.append(point[0])
            npoints.append(point[-1])
    return torch.stack(npoints)


def get_first_and_last(x):
    return torch.concat([x[:, :2], x[:, -2:]], dim=-1)


def _dist_each(x, npoints):
    return torch.sqrt(((x[:, 2:] - x[:, :-2]) ** 2).reshape(-1, 2).sum(-1)).reshape(-1, npoints - 1)


def line_straightness(x):
    n = int(x.shape[-1] / 2)
    dist_ends = torch.sqrt(((x[:, :2] - x[:, -2:]) ** 2).sum(-1))
    dist_each = _dist_each(x, n).sum(-1)
    return dist_each / dist_ends


def point_spacing_std(x):
    n = int(x.shape[-1] / 2)
    d = _dist_each(x, n)
    if d.shape[-1] > 1:
        return d.std(-1)
    else:
        return torch.zeros(d.shape[0]).to(d.device)


def _dist(x, y):
    """
    Manhattan distance to keep the scale but to avoid nan gradients at 0 for sqrt (in Cartesian distance)

    """
    return torch.abs(x - y).sum(-1)


def point_segment_dist(v, w, p):
    """
    Pytorch implementation of this solution: https://stackoverflow.com/a/1501725/13120052
    """
    l2 = ((w - v) ** 2).sum(-1)  # distance squared
    t = ((p - v) * (w - v)).sum(-1) / (l2 + 10 ** (-10))  # relative projection of point p onto the segment
    t = torch.stack([torch.ones(len(t)).to(t.device), t]).min(0)[0]  # clamp the projection
    t = torch.stack([torch.zeros(len(t)).to(t.device), t]).max(0)[0]
    proj = v + t.unsqueeze(-1) * (w - v)
    return _dist(p, proj)
