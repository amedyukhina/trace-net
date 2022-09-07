import os
from pathlib import Path

from torch.utils.data import DataLoader

from tracenet.datasets.filament import FilamentDetection
from tracenet.datasets.transforms import (
    get_train_transform,
    get_valid_transform,
    collate_fn
)
from tracenet.datasets.transforms_segm import get_valid_transform_segm, get_train_transform_segm


def get_loaders(data_dir, img_dir='img', gt_dir='gt', train_dir='train', val_dir='val',
                train_transform=None, valid_transform=None, dataset=None,
                maxsize=None, n_points=2, batch_size=2, **_):
    if dataset is None:
        dataset = FilamentDetection
        ext = '.csv'
        default_train_tranform, default_val_trainform = (get_train_transform(), get_valid_transform())
    else:
        ext = '.tif'
        kw = dict(patch_size=maxsize) if maxsize is not None else dict()
        default_train_tranform, default_val_trainform = (get_train_transform_segm(**kw),
                                                         get_valid_transform_segm(**kw))

    # Get transforms
    if train_transform is None:
        train_transform = default_train_tranform
    if valid_transform is None:
        valid_transform = default_val_trainform
    transforms = [train_transform, valid_transform]

    # Get datasets
    data_dir = Path(data_dir)
    ds = []
    for dset, transform in zip([train_dir, val_dir], transforms):
        files = [fn for fn in os.listdir(data_dir / dset / img_dir) if fn.endswith('.tif')]
        files.sort()
        ds.append(
            dataset(
                [data_dir / dset / img_dir / fn for fn in files],
                [data_dir / dset / gt_dir / fn.replace('.tif', ext) for fn in files],
                maxsize=maxsize, n_points=n_points,
                transforms=transform
            )
        )
    ds_train, ds_val = ds

    # Get loaders
    dl_train = DataLoader(ds_train, shuffle=True,
                          collate_fn=collate_fn,
                          batch_size=batch_size, num_workers=batch_size)
    dl_val = DataLoader(ds_val, shuffle=False,
                        collate_fn=collate_fn,
                        batch_size=batch_size, num_workers=batch_size)
    return dl_train, dl_val
