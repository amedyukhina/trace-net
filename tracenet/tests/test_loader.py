import numpy as np
import pytest
import torch

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


@pytest.fixture(params=[0.1, 0.5, 0.8, 0.9])
def instance_ratio(request):
    return request.param


def test_loader(example_data_path, dl_index):
    loader = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1)[dl_index]
    imgs1, imgs2, targets, labels, masks = get_batch_and_assert(loader)

    assert isinstance(targets, tuple)
    assert len(targets) == masks.shape[0]
    assert 'boxes' in targets[0].keys()
    assert isinstance(targets[0]['boxes'], torch.Tensor)
    assert len(targets[0]['boxes'].shape) == 2


def test_loader_segm(example_segm_data_path, dl_index, random_size):
    loader = get_loaders(example_segm_data_path, train_dir='', val_dir='', batch_size=1,
                         train_transform=get_train_transform_segm(random_size),
                         valid_transform=get_valid_transform_segm(random_size),
                         segm_only=True)[dl_index]
    imgs1, imgs2, _, labels, masks = get_batch_and_assert(loader)
    assert masks.shape[-2:] == (random_size, random_size)


def test_spoco_loader(example_segm_data_path, instance_ratio):
    train_dl, val_dl = get_loaders(example_segm_data_path, train_dir='', val_dir='', batch_size=1,
                                   train_transform=get_valid_transform_segm(500),
                                   segm_only=True, instance_ratio=instance_ratio, maxsize=500)
    _, _, _, labels_train, _ = get_batch_and_assert(train_dl)
    _, _, _, labels_val, _ = get_batch_and_assert(val_dl)
    assert len(np.unique(labels_train)[1:]) == int(round(instance_ratio * len(np.unique(labels_val)[1:])))


def get_batch_and_assert(loader):
    batch = next(iter(loader))
    assert len(batch) == 5
    imgs1, imgs2, targets, labels, masks = batch
    for imgs in [imgs1, imgs2]:
        assert isinstance(imgs, torch.Tensor)
        assert imgs.max() == 1
        assert imgs.min() == 0
    assert isinstance(labels, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert imgs1.shape[0] == imgs2.shape[0] == labels.shape[0] == masks.shape[0]
    assert len(imgs1.shape) == len(imgs2.shape) == len(labels.shape) + 1 == len(masks.shape) + 1
    assert imgs1.shape[-2:] == imgs2.shape[-2:] == labels.shape[-2:] == masks.shape[-2:]
    return imgs1, imgs2, targets, labels, masks
