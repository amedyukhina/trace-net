import numpy as np
import pytest
import torch

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
    get_batch_and_assert(loader)


# def test_spoco_loader(example_segm_data_path, instance_ratio):
#     train_dl, val_dl = get_loaders(example_segm_data_path, train_dir='', val_dir='', batch_size=1,
#                                    train_transform=get_valid_transform_segm(500),
#                                    segm_only=True, instance_ratio=instance_ratio, maxsize=500)
#     _, _, _, labels_train, _ = get_batch_and_assert(train_dl)
#     _, _, _, labels_val, _ = get_batch_and_assert(val_dl)
#     assert len(np.unique(labels_train)[1:]) == int(round(instance_ratio * len(np.unique(labels_val)[1:])))


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

    for key in ['keypoints', 'point_labels']:
        assert key in targets
        assert isinstance(targets[key], list)
        assert len(targets[key]) == imgs1.shape[0]

    assert len(imgs1.shape) == len(imgs2.shape) == len(targets['mask'].shape) + 1 \
           == len(targets['labeled_mask'].shape) + 1
    assert imgs1.shape[-2:] == imgs2.shape[-2:] == targets['mask'].shape[-2:] == targets['labeled_mask'].shape[-2:]
    return imgs1, imgs2, targets
