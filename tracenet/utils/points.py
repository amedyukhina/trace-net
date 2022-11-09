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
