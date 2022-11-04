import os
from pathlib import Path

from torch.utils.data import DataLoader

from tracenet.datasets.filament import Filament
from tracenet.datasets.transforms import (
    get_train_transform,
    get_intensity_transform,
    collate_fn
)


def get_loaders(data_dir, img_dir='img', gt_dir='gt', train_dir='train', val_dir='val',
                train_transform=None, valid_transform=None, intensity_transform=None, shuffle=True,
                maxsize=512, n_points=2, batch_size=2, instance_ratio=1, mean_std=(0, 1), dataset=None, **_):
    # Get Transforms
    if dataset is None:
        dataset = Filament
    transforms = [
        dict(transforms=get_train_transform() if train_transform is None else train_transform,
             intensity_transforms=get_intensity_transform() if intensity_transform is None
             else intensity_transform,
             instance_ratio=instance_ratio),
        dict(transforms=valid_transform,
             intensity_transforms=None,
             instance_ratio=1)
    ]

    # Get datasets
    data_dir = Path(data_dir)
    ds = []
    for dset, transform in zip([train_dir, val_dir], transforms):
        files = [fn for fn in os.listdir(data_dir / dset / img_dir) if fn.endswith('.tif')]
        files.sort()
        ds.append(
            dataset(
                [data_dir / dset / img_dir / fn for fn in files],
                [data_dir / dset / gt_dir / fn.replace('.tif', '.csv') for fn in files],
                maxsize=maxsize, mean_std=mean_std, n_points=n_points, **transform
            )
        )
    ds_train, ds_val = ds

    # Get loaders
    dl_train = DataLoader(ds_train, shuffle=shuffle,
                          collate_fn=collate_fn,
                          batch_size=batch_size, num_workers=batch_size)
    dl_val = DataLoader(ds_val, shuffle=False,
                        collate_fn=collate_fn,
                        batch_size=batch_size, num_workers=batch_size)
    return dl_train, dl_val
