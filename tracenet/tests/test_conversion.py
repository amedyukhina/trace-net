import numpy as np

from ..utils.points import (
    normalize_points,
    denormalize_points
)


def test_normalization(random_imgsize, random_points):
    points, _ = random_points
    point_converted = denormalize_points(normalize_points(points, random_imgsize), random_imgsize).numpy()
    point_converted = np.round_(point_converted)
    assert (points.numpy() == point_converted).all()
