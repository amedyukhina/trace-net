import os
from pathlib import Path

from torch.utils.data import DataLoader

from tracenet.datasets import FilamentDetection
from tracenet.datasets.transforms import (
    get_train_transform_intensity,
    get_train_transform_spatial,
    get_valid_transform,
    collate_fn
)


def get_loaders(data_dir, img_dir='img', gt_dir='gt', train_dir='train', val_dir='val',
                train_transforms=None, valid_transform=None,
                maxsize=None, n_points=2, batch_size=2, **_):
    # Get transforms
    if train_transforms is None:
        train_transforms = [get_train_transform_spatial(), get_train_transform_intensity()]
    if valid_transform is None:
        valid_transform = get_valid_transform()
    transforms = [dict(spatial_transforms=train_transforms[0],
                       intensity_transforms=train_transforms[1]),
                  dict(spatial_transforms=valid_transform)]

    # Get datasets
    data_dir = Path(data_dir)
    ds = []
    for dset, transform in zip([train_dir, val_dir], transforms):
        files = [fn for fn in os.listdir(data_dir / dset / img_dir) if fn.endswith('.tif')]
        files.sort()
        ds.append(
            FilamentDetection(
                [data_dir / dset / img_dir / fn for fn in files],
                [data_dir / dset / gt_dir / fn.replace('.tif', '.csv') for fn in files],
                maxsize=maxsize, n_points=n_points,
                **transform
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
