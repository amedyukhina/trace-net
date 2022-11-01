import numpy as np
import pytest
import torch
from monai.networks.nets import UNet

from tracenet.losses.cldice import SoftClDice, SoftDiceClDice
from tracenet.losses.contrastive import expand_as_one_hot
from tracenet.losses.indexing import intensity_loss, dist_push
from tracenet.utils.loader import get_loaders


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


def test_cldice(example_data_path):
    loader = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1, mean_std=(3, 0.4))[0]
    imgs, _, targets = next(iter(loader))
    net = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=2,
        channels=(8, 16, 64),
        strides=(2, 2),
        num_res_units=1,
        dropout=0.1
    ).cuda()
    outputs = net(imgs.cuda())
    for loss_fn in [SoftClDice(), SoftDiceClDice(alpha=0.5, include_background=True,
                                                 to_onehot_y=True, softmax=False)]:
        loss = loss_fn(outputs, targets['mask'].cuda())
        assert loss.item() > 0
        loss = loss_fn(expand_as_one_hot(targets['mask'], 2), targets['mask'])
        assert loss == 0
