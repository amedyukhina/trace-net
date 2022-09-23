import torch


def normalize_points(x, size):
    return x / torch.tensor(size)


def denormalize_points(x, size):
    return x * torch.tensor(size)


def coord_to_al(x1, y1, x2, y2):
    vec = torch.stack([y2 - y1, x2 - x1], dim=-1)
    l = torch.norm(vec, dim=-1)
    y, x = vec.unbind(-1)
    a = torch.atan(y / x) + torch.pi / 2 * (1 - torch.sign(x))
    return a, l


def al_to_coord(x1, y1, a, l):
    x2 = x1 + l * torch.cos(a)
    y2 = y1 + l * torch.sin(a)
    return x2, y2


def points_to_bounding_line(x):
    points = x.reshape(-1, 4).unbind(-1)
    y1, x1 = points[:2]
    y2, x2 = points[-2:]
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    a, l = coord_to_al(x1, y1, x2, y2)
    a = a + torch.pi / 2
    a[a < 0] = a[a < 0] + torch.pi
    a = a / torch.pi
    b = [y_c, x_c, l, a]
    return torch.stack(b, dim=-1)


def bounding_line_to_points(x):
    y_c, x_c, l, a = x.unbind(-1)
    a = a * torch.pi - torch.pi / 2
    x_d = l / 2 * torch.cos(a)
    y_d = l / 2 * torch.sin(a)
    b = [y_c - y_d, x_c - x_d, y_c + y_d, x_c + x_d]
    return torch.stack(b, dim=-1).reshape(-1, 2)


def get_first_and_last_points(points, labels):
    npoints = []
    for lb in labels.unique():
        point = points[labels == lb]
        if len(points) > 1:
            npoints.append(point[0])
            npoints.append(point[-1])
    return torch.stack(npoints)
