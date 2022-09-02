import numpy as np
import pandas as pd
import torch
import torch.utils.data
from skimage import io

from .transforms import apply_transform, norm_pad_to_gray, pad_to_max, get_valid_transform
from ..utils.points import normalize_points, points_to_bounding_line


class FilamentDetection(torch.utils.data.Dataset):
    def __init__(self, image_files, ann_files,
                 spatial_transforms=None, intensity_transforms=None,
                 col_id='id', maxsize=None, n_points=10, cols=None):
        self.intensity_transforms = intensity_transforms
        self.spatial_transforms = spatial_transforms
        if spatial_transforms is None:
            self.spatial_transforms = get_valid_transform()
        self.ann_files = ann_files
        self.image_files = image_files
        self.col_id = col_id
        self.maxsize = maxsize
        self.n_points = n_points
        self.cols = cols if cols is not None else ['x', 'y']

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = norm_pad_to_gray(io.imread(image_id), maxsize=self.maxsize)

        df = pd.read_csv(ann_id)
        mask = generate_labeled_mask(df, image.shape, ['y', 'x'], n_interp=30, id_col=self.col_id)
        image = np.dstack([image, mask, (mask > 0) * 1])

        boxes = []
        for s in df[self.col_id].unique():
            cur_df = df[df[self.col_id] == s].reset_index(drop=True)
            # if len(cur_df[self.cols].values.ravel()) != 2 * self.n_points:
            #     cur_df = make_points_equally_spaced(cur_df, self.cols, n_points=self.n_points)
            coords = cur_df[self.cols].values.ravel()
            boxes.append(coords)

        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        target = self.set_target(boxes, index)
        image, target, label, mask = self.apply_transforms(image, target)

        return image, target, label, mask

    def set_target(self, boxes, index):
        area = torch.ones((boxes.shape[0],), dtype=torch.float32)
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        point_labels = list(np.arange((boxes.shape[-1] // 2)).astype(str)) * boxes.shape[0]

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([index]),
            area=area,
            iscrowd=iscrowd,
            point_labels=point_labels
        )
        return target

    def apply_transforms(self, image, target):
        if self.spatial_transforms:
            target, image = apply_transform(self.spatial_transforms, target, image)
        image, label, mask = image
        image = torch.stack([image] * 3)
        if self.intensity_transforms:
            target, image = apply_transform(self.intensity_transforms, target, np.moveaxis(image.numpy(), 0, -1))
        target['boxes'] = normalize_points(target['boxes'], image.shape[-2:])
        target['boxes'] = points_to_bounding_line(target['boxes'])
        target['area'] = torch.ones((target['boxes'].shape[0],), dtype=torch.float32)
        return image, target, label, mask

    def __len__(self) -> int:
        return len(self.image_files)


class FilamentSegmentation(FilamentDetection):

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = norm_pad_to_gray(io.imread(image_id), maxsize=self.maxsize)
        mask = pad_to_max(io.imread(ann_id), maxsize=self.maxsize)
        image = np.dstack([image, mask, (mask > 0) * 1])

        boxes = torch.zeros((1, self.n_points * 2), dtype=torch.float32)
        target = self.set_target(boxes, index)
        image, target, label, mask = self.apply_transforms(image, target)
        return image, target, label, mask

    def __len__(self) -> int:
        return len(self.image_files)


def make_points_equally_spaced(df, cols, n_points=10):
    """
    Interpolate the filament to a given number of points.
    """
    df = _interpolate_points(df.copy(), cols, n_interp=3)

    # create empty image
    img = np.zeros(np.array([int(df[c].max()) + 10 for c in cols]))

    # calculate the distance profile
    coords = df[cols].values
    dist = np.array([0] + list(_dist(coords[:-1], coords[1:])))
    dist = np.cumsum(dist)
    dist_img = np.zeros_like(img)
    dist_img[tuple(np.int_(coords.transpose()))] = dist

    # new distance profile
    n_dist = np.linspace(dist[0], dist[-1], n_points)

    # new coordinates
    n_coords = [coords[0]]
    for i in range(1, len(n_dist)):
        diff_img = np.abs(dist_img - n_dist[i])
        n_coords.append(np.array(np.where(diff_img == np.min(diff_img))).transpose()[0])

    n_df = pd.DataFrame(n_coords, columns=cols)
    return n_df


def generate_labeled_mask(df, shape, cols, n_interp=30, id_col='id'):
    df_interp = _interpolate_points(df, n_interp=n_interp, cols=cols, id_col=id_col)
    mt_img = _create_mt_image(df_interp, shape=shape, cols=cols)
    return mt_img


def _dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def _interpolate_points(df, cols, n_interp=10, id_col='id'):
    new_df = pd.DataFrame()
    for s in df[id_col].unique():
        coords = df[df[id_col] == s][cols].values
        for i in range(len(coords) - 1):
            new_coords = np.array([np.linspace(coords[i, j], coords[i + 1, j], n_interp, endpoint=False)
                                   for j in range(len(coords[i]))]).transpose()
            cur_df = pd.DataFrame(new_coords, columns=cols)
            cur_df[id_col] = s
            new_df = pd.concat([new_df, cur_df], ignore_index=True)
    return new_df


def _create_mt_image(df, shape, cols, id_col='id'):
    img = np.zeros(shape)
    if len(df) > 0:
        for mt_id in df[id_col].unique():
            cur_points = df[df[id_col] == mt_id]
            coords = [np.int_(cur_points[c]) for c in cols]
            img[tuple(coords)] = mt_id
    return img
