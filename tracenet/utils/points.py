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


def point_order_loss(x):
    n = int(x.shape[-1] / 2)
    dist_ends = torch.sqrt(((x[:, :2] - x[:, -2:]) ** 2).sum(-1))
    dist_each = _dist_each(x, n).sum(-1)
    return dist_each / dist_ends


def point_spacing_loss(x):
    n = int(x.shape[-1] / 2)
    return _dist_each(x, n).std(-1)


def _dist(x, y):
    return torch.sqrt(((x - y) ** 2).sum(-1))


def dist_point_segment(v, w, p):
    """
    Pytorch implementation of this solution: https://stackoverflow.com/a/1501725/13120052
    """
    l2 = _dist(v, w)
    t = ((p - v) * (w - v)).sum(-1) / l2
    t = torch.stack([torch.ones(len(t)), t]).min(0)[0]
    t = torch.stack([torch.zeros(len(t)), t]).max(0)[0]
    proj = v + t.unsqueeze(-1) * (w - v)
    return _dist(p, proj)
