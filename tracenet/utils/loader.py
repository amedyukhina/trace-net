import os
from pathlib import Path

from torch.utils.data import DataLoader

from tracenet.datasets.filament import FilamentDetection, FilamentSegmentation
from tracenet.datasets.transforms import (
    get_train_transform,
    get_valid_transform,
    get_intensity_transform,
    collate_fn
)
from tracenet.datasets.transforms_segm import (
    get_valid_transform_segm,
    get_train_transform_segm,
)


def get_loaders(data_dir, img_dir='img', gt_dir='gt', train_dir='train', val_dir='val',
                train_transform=None, valid_transform=None, intensity_transform=None,
                segm_only=False, maxsize=None, n_points=2, batch_size=2, instance_ratio=1, **_):
    # Get Transforms
    if segm_only:
        dataset = FilamentSegmentation
        ext = '.tif'
        kw = dict(patch_size=maxsize) if maxsize is not None else dict()
        transforms = [
            dict(transforms=get_train_transform_segm(**kw) if train_transform is None else train_transform,
                 intensity_transforms=get_intensity_transform() if intensity_transform is None
                 else intensity_transform,
                 instance_ratio=instance_ratio),
            dict(transforms=get_valid_transform_segm(**kw) if valid_transform is None else valid_transform,
                 intensity_transforms=None,
                 instance_ratio=1)
        ]

    else:
        dataset = FilamentDetection
        ext = '.csv'
        transforms = [
            dict(transforms=get_train_transform() if train_transform is None else train_transform,
                 intensity_transforms=get_intensity_transform() if intensity_transform is None
                 else intensity_transform,
                 instance_ratio=instance_ratio),
            dict(transforms=get_valid_transform() if valid_transform is None else valid_transform,
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
                [data_dir / dset / gt_dir / fn.replace('.tif', ext) for fn in files],
                maxsize=maxsize, n_points=n_points, **transform
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
