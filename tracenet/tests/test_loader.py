import albumentations as A
import numpy as np
import pytest
import torch

from tracenet.datasets.transforms import KEYPOINT_PARAMS
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
    loader = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1, mean_std=(3, 0.4))[dl_index]
    get_batch_and_assert(loader)


def test_spoco_loader(example_data_path, instance_ratio):
    transform = A.Compose([], keypoint_params=KEYPOINT_PARAMS)
    train_dl, val_dl = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1,
                                   train_transform=transform,
                                   instance_ratio=instance_ratio, shuffle=False)
    _, _, target_train = get_batch_and_assert(train_dl)
    _, _, target_val = get_batch_and_assert(val_dl)
    assert len(np.unique(target_train['point_labels'][0])) == \
           int(round(len(np.unique(target_val['point_labels'][0])) * instance_ratio))
    assert len(np.unique(target_train['labeled_mask'])[1:]) == \
           int(round(instance_ratio * len(np.unique(target_val['labeled_mask'])[1:])))


def get_batch_and_assert(loader):
    batch = next(iter(loader))
    assert len(batch) == 3
    imgs1, imgs2, targets = batch
    for imgs in [imgs1, imgs2]:
        assert isinstance(imgs, torch.Tensor)
    assert imgs1.shape[0] == imgs2.shape[0]
    assert isinstance(targets, dict)
    for key in ['mask', 'labeled_mask', 'padding']:
        assert key in targets
        assert isinstance(targets[key], torch.Tensor)
        assert targets[key].shape[0] == imgs1.shape[0]

    for key in ['keypoints', 'point_labels', 'trace', 'trace_class']:
        assert key in targets
        assert isinstance(targets[key], list)
        assert len(targets[key]) == imgs1.shape[0]

    assert len(imgs1.shape) == len(imgs2.shape) == len(targets['mask'].shape) + 1 \
           == len(targets['labeled_mask'].shape) + 1
    assert imgs1.shape[-2:] == imgs2.shape[-2:] == targets['mask'].shape[-2:] == targets['labeled_mask'].shape[-2:]
    return imgs1, imgs2, targets
