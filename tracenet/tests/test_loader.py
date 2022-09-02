import torch

from tracenet.utils.loader import get_loaders


def test_loader(example_data_path):
    train_dl, val_dl = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1)
    imgs, targets, labels, masks = next(iter(train_dl))
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
