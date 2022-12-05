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


def _dist_each_mh(x, npoints):
    return (torch.abs(x[:, 2:] - x[:, :-2]).reshape(-1, 2).sum(-1)).reshape(-1, npoints - 1)


def line_straightness_mh(x):
    n = int(x.shape[-1] / 2)
    dist_ends = torch.abs(x[:, :2] - x[:, -2:]).sum(-1)
    dist_each = _dist_each_mh(x, n).sum(-1)
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
    Manhattan distance to keep the scale but to avoid nan gradients at 0 for sqrt

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


def trace_distance(src_traces, target_traces):
    x = torch.stack([target_traces[:, 2 * i:2 * i + 4]
                     for i in range(int(target_traces.shape[-1] / 2) - 1)])  # get all segments of the target
    v = x[:, :, :2]  # first points of all target segments
    w = x[:, :, 2:]  # second points of all target segments
    points = [src_traces[:, 2 * j:2 * j + 2]
              for j in range(int(src_traces.shape[-1] / 2))]  # all points of the source trace
    trace_dist = torch.stack([
        torch.stack([point_segment_dist(vv, ww, p)
                     for vv, ww in zip(v, w)]).min(0)[0]  # dist from each point to the closest segment
        for p in points]).mean(0)  # average among all points
    return trace_dist


def trace_distance_param(src_traces, target_traces):
    trace_dist = torch.stack([torch.abs(target_traces[:, i].unsqueeze(1) - src_traces).sum(-1).min(-1)[0]
                              for i in range(target_traces.shape[1])]).mean(0)
    return trace_dist


def bezier_curve_from_control_points(cpoints, n_points=10):
    t = torch.linspace(0, 1, n_points).unsqueeze(1).unsqueeze(0).to(cpoints.device)
    b = (1 - t) ** 3 * cpoints[:, 0].unsqueeze(1) + 3 * (1 - t) ** 2 * t * cpoints[:, 1].unsqueeze(1) + \
        3 * (1 - t) * t ** 2 * cpoints[:, 2].unsqueeze(1) + t ** 3 * cpoints[:, 3].unsqueeze(1)
    return b
