import numpy as np

from ..utils.points import (
    normalize_points,
    denormalize_points,
    points_to_bounding_line,
    bounding_line_to_points,
    get_first_and_last_points
)


def test_normalization(random_imgsize, random_points):
    points, _ = random_points
    point_converted = denormalize_points(normalize_points(points, random_imgsize), random_imgsize).numpy()
    point_converted = np.round_(point_converted)
    assert (points.numpy() == point_converted).all()


def test_conversion(random_imgsize, random_points):
    points, labels = random_points
    points_converted = points_to_bounding_line(
        get_first_and_last_points(
            normalize_points(points, random_imgsize),
            labels
        )
    )
    assert len(points_converted.shape) == 2
    assert points_converted.shape[-1] == 4
    points_back_converted = denormalize_points(bounding_line_to_points(points_converted), random_imgsize)
    points_back_converted = np.round_(points_back_converted.numpy())
    points = get_first_and_last_points(points, labels)
    assert len(points) == 2 * len(labels.unique())
    points = points.numpy()
    assert (points_back_converted == points).all()
    assert (points_back_converted == points).all()
