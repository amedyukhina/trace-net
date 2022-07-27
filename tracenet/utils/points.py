import torch

from .utils import angle_between


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
    a = angle_between([y2 - y1, x2 - x1], [0, 1])
    return a, l


def one_al_to_coord(x1, y1, a, l):
    x2 = x1 + l * torch.cos(a)
    y2 = y1 + l * torch.sin(a)
    return x2, y2
