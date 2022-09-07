import numpy as np
import pytest
import torch

from tracenet.datasets.filament import FilamentSegmentation
from tracenet.datasets.transforms_segm import get_valid_transform_segm, get_train_transform_segm
from tracenet.utils.loader import get_loaders


@pytest.fixture(params=[
    0,
    1
])
def dl_index(request):
    return request.param


@pytest.fixture(params=np.random.randint(10, 120, 10))
def random_size(request):
    return request.param


def test_loader(example_data_path, dl_index):
    loader = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1)[dl_index]
    imgs, targets, labels, masks = next(iter(loader))
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert isinstance(targets, tuple)
    assert imgs.shape[0] == labels.shape[0] == masks.shape[0] == len(targets)
    assert imgs.max() <= 1
    assert len(imgs.shape) == len(labels.shape) + 1 == len(masks.shape) + 1
    assert 'boxes' in targets[0].keys()
    assert isinstance(targets[0]['boxes'], torch.Tensor)
    assert imgs.shape[-2:] == labels.shape[-2:] == masks.shape[-2:]
    assert len(targets[0]['boxes'].shape) == 2


def test_loader_segm(example_segm_data_path, dl_index, random_size):
    loader = get_loaders(example_segm_data_path, train_dir='', val_dir='', batch_size=1,
                         train_transform=get_train_transform_segm(random_size),
                         valid_transform=get_valid_transform_segm(random_size),
                         dataset=FilamentSegmentation)[dl_index]
    imgs, _, labels, masks = next(iter(loader))
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert imgs.shape[0] == labels.shape[0] == masks.shape[0]
    assert imgs.max() <= 1
    assert len(imgs.shape) == len(labels.shape) + 1 == len(masks.shape) + 1
    assert imgs.shape[-2:] == labels.shape[-2:] == masks.shape[-2:] == (random_size, random_size)
