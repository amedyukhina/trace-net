import pytest
import torch

from tracenet.datasets.filament import FilamentSegmentation
from tracenet.utils.loader import get_loaders


@pytest.fixture(params=[
    None,
    # FilamentSegmentation
])
def datasets(request, example_data_path, example_segm_data_path):
    return request.param, example_data_path if request.param is None else example_segm_data_path


@pytest.fixture(params=[
    0,
    1
])
def loader(request, datasets):
    dataset, path = datasets
    loaders = get_loaders(path, train_dir='', val_dir='', batch_size=1, dataset=dataset)
    return loaders[request.param], path


def test_loader(loader):
    loader, path = loader
    imgs, targets, labels, masks = next(iter(loader))
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert isinstance(targets, tuple)
    assert imgs.shape[0] == labels.shape[0] == masks.shape[0] == len(targets)
    assert len(imgs.shape) == len(labels.shape) + 1 == len(masks.shape) + 1
    assert 'boxes' in targets[0].keys()
    assert isinstance(targets[0]['boxes'], torch.Tensor)
    assert imgs.shape[-2:] == labels.shape[-2:] == masks.shape[-2:]
    assert len(targets[0]['boxes'].shape) == 2
