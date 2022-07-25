import numpy as np

from ..utils import cxcywh_to_xyxy, xyxy_to_cxcywh, normalize_points, denormalize_points


def test_bbox_conversion(random_imgsize, random_bboxes):
    bboxes_converted = cxcywh_to_xyxy(xyxy_to_cxcywh(random_bboxes))
    assert (random_bboxes == bboxes_converted).all()

    point_converted = denormalize_points(normalize_points(random_bboxes, random_imgsize), random_imgsize).numpy()
    point_converted = np.round_(point_converted)
    assert (random_bboxes.numpy() == point_converted).all()


def test_normalization(random_imgsize, random_points):
    point_converted = denormalize_points(normalize_points(random_points, random_imgsize), random_imgsize).numpy()
    point_converted = np.round_(point_converted)
    assert (random_points.numpy() == point_converted).all()
