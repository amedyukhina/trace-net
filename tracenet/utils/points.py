import torch


def xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def normalize_points(x, size):
    ndim = x.shape[-1]
    x = x.reshape(-1, 2)
    x = x / torch.tensor(size)
    return x.reshape(-1, ndim)


def denormalize_points(x, size):
    ndim = x.shape[-1]
    x = x.reshape(-1, 2)
    x = x * torch.tensor(size)
    return x.reshape(-1, ndim).to(float)


def one_coord_to_al(x1, y1, x2, y2):
    l = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    a = torch.atan((y2 - y1) / (x2 - x1))
    return a, l


def one_al_to_coord(x1, y1, a, l):
    x2 = x1 + l * torch.cos(a)
    y2 = y1 + l * torch.sin(a)
    return x2, y2


def points_to_bounding_line(x):
    points = x.unbind(-1)
    y1, x1 = points[:2]
    y2, x2 = points[-2:]
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    vec = torch.stack([y2 - y1, x2 - x1], dim=-1)
    l = torch.norm(vec, dim=-1)
    a = (torch.atan(vec[:, 0] / vec[:, 1]) + torch.pi/2) / torch.pi
    b = [y_c, x_c, l, a]
    return torch.stack(b, dim=-1)


def bounding_line_to_points(x):
    y_c, x_c, l, a = x.unbind(-1)
    a = a * torch.pi - torch.pi/2
    x_d = l / 2 * torch.cos(a)
    y_d = l / 2 * torch.sin(a)
    b = [y_c - y_d, x_c - x_d, y_c + y_d, x_c + x_d]
    return torch.stack(b, dim=-1)
