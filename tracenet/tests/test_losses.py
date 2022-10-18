import numpy as np
import pytest
import torch

from tracenet.losses.indexing import intensity_loss, dist_push


@pytest.fixture(params=[5, 10, 100])
def trace(request):
    return torch.tensor(np.array([np.linspace(0, 1, request.param, endpoint=False),
                                  np.linspace(0, 1, request.param, endpoint=False)]).transpose()), request.param


def test_dist_push(trace):
    trace, npoints = trace
    dist = dist_push(trace, mindist=1. / npoints / 2)
    assert dist == 0
    dist = dist_push(trace, mindist=1. / npoints * 2)
    assert dist > 0


def test_diff_ind(trace):
    img = torch.ones([50, 50]) * 255
    assert intensity_loss(img, trace[0], maxval=255) == 0
    assert intensity_loss(img, trace[0], maxval=500) >= 0
