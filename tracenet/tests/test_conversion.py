import numpy as np

from ..utils.points import normalize_points, denormalize_points, points_to_bounding_line, bounding_line_to_points


def test_normalization(random_imgsize, random_points):
    point_converted = denormalize_points(normalize_points(random_points, random_imgsize), random_imgsize).numpy()
    point_converted = np.round_(point_converted)
    assert (random_points.numpy() == point_converted).all()


def test_conversion(random_imgsize, random_points):
    points_converted = points_to_bounding_line(normalize_points(random_points, random_imgsize))
    points_back_converted = denormalize_points(bounding_line_to_points(points_converted), random_imgsize)
    points_back_converted = np.round_(points_back_converted.numpy())
    random_points = random_points.numpy()
    assert (points_back_converted[:, :2] == random_points[:, :2]).all()
    assert (points_back_converted[:, -2:] == random_points[:, -2:]).all()

