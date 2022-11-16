import numpy as np
import pytest
import torch

from tracenet.utils.points import point_segment_dist


@pytest.fixture(params=[0.5, 5, 10])
def len_segm(request):
    return request.param


def test_point_on_segment():
    x = np.random.randint(0, 100, (2, 10))
    y = np.linspace(x[0], x[1], 3)[1]
    v = torch.tensor(x[0].reshape(-1, 2))
    w = torch.tensor(x[1].reshape(-1, 2))
    p = torch.tensor(y.reshape(-1, 2), requires_grad=True)
    loss = point_segment_dist(v, w, p)
    assert (loss == 0).all()
    loss.sum().backward()
    assert (p.grad == 0).all()


def test_point_outside(len_segm):
    v = torch.tensor([[0, 0]])
    w = torch.tensor([[0, len_segm]])
    p = torch.tensor([[1, -1]])
    assert point_segment_dist(v, w, p).item() == 2
    p = torch.tensor([[1, len_segm + 1]])
    assert point_segment_dist(v, w, p).item() == 2
