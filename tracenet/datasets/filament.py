import numpy as np
import pandas as pd
import torch
import torch.utils.data
from skimage import io

from tracenet.datasets.transforms import apply_transform
from ..utils.points import (
    normalize_points,
    points_to_bounding_line,
    get_first_and_last_points
)


class Filament(torch.utils.data.Dataset):
    """
    Dataset for the filament detection and segmentation.

    The Dataset expects as input:
        - 2D images of size <= 512 pixels, pixel size 0.035 microns (18 microns image size)
        - List of x and y coordinates of the filaments as a csv file (include "x", "y" and "id" columns).
    """

    def __init__(self, image_files, ann_files, mean_std=(0, 1), transforms=None, intensity_transforms=None,
                 instance_ratio=1, col_id='id', maxsize=512, n_points=2, cols=None):
        self.transforms = transforms
        self.intensity_transforms = intensity_transforms
        self.instance_ratio = instance_ratio
        self.ann_files = ann_files
        self.image_files = image_files
        self.seeds = np.random.randint(0, np.iinfo('int32').max, len(image_files))
        self.col_id = col_id
        self.maxsize = maxsize
        self.n_points = n_points
        self.cols = cols if cols is not None else ['y', 'x']
        self.mean, self.std = mean_std

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = io.imread(image_id).astype(np.float32)
        if len(image.shape) > 2:
            image = np.max(image, axis=-1)
        image = (image - self.mean) / self.std

        if np.max(image.shape) > self.maxsize:
            raise ValueError(rf"Image size must be less than or equal to {self.maxsize};"
                             rf"current image shape is {image.shape}")
        image, padding = pad_to_size(image, self.maxsize)
        image = np.dstack([image] * 3)

        df = pd.read_csv(ann_id)

        if self.instance_ratio < 1:
            df = sample_instances(df, self.instance_ratio, seed=self.seeds[index], col_id=self.col_id)

        points, labels = df_to_points(df, self.cols, self.col_id)
        target = dict(
            keypoints=points + padding,
            point_labels=labels,
        )

        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)

        mask = torch.tensor(generate_labeled_mask(target['keypoints'],
                                                  target['point_labels'],
                                                  image.shape[:2], n_interp=30),
                            dtype=torch.int64)
        mask = torch.unique(mask, return_inverse=True)[1].reshape(mask.shape)

        target['mask'] = (mask > 0) * 1
        target['labeled_mask'] = mask
        target['padding'] = torch.tensor(padding, dtype=torch.float64)
        target['keypoints'] = torch.tensor(target['keypoints'], dtype=torch.float64)
        target['point_labels'] = torch.tensor(target['point_labels'], dtype=torch.int64)

        target['trace'] = points_to_bounding_line(
            get_first_and_last_points(
                normalize_points(target['keypoints'], image.shape[:2]),
                target['point_labels']
            )
        ).float()
        target['trace_class'] = torch.ones((target['trace'].shape[0],), dtype=torch.int64)

        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float64)

        if self.intensity_transforms:
            image1 = self.transform_intensity(image)
            image2 = self.transform_intensity(image)
        else:
            image1 = image2 = image

        return image1.float(), image2.float(), target

    def transform_intensity(self, image):
        image2 = self.intensity_transforms(image)
        if image2.max() > 0:
            return image2
        else:
            return self.transform_intensity(image)

    def __len__(self) -> int:
        return len(self.image_files)


def sample_instances(df, instance_ratio, seed, col_id, min_instances=2):
    llist = df[col_id].unique()
    n_objects = int(max(min_instances, round(instance_ratio * len(llist))))
    np.random.seed(seed)
    np.random.shuffle(llist)
    df_new = df[df[col_id].isin(llist[:n_objects])].reset_index(drop=True)
    return df_new


def pad_to_size(image, size=512):
    diff = size - np.array(image.shape)
    pad_left = np.int_(diff / 2)
    pad_right = diff - pad_left
    image = np.pad(image, [(pad_left[0], pad_right[0]),
                           (pad_left[1], pad_right[1])])
    return image, pad_left


def df_to_points(df, cols, col_id):
    points = []
    labels = []
    for s in df[col_id].unique():
        cur_df = df[df[col_id] == s].reset_index(drop=True)
        coords = cur_df[cols].values
        points.append(coords)
        labels.append([s + 1] * len(coords))

    return np.concatenate(points, axis=0), np.ravel(labels)


def generate_labeled_mask(points, labels, shape, n_interp=30):
    points, labels = _interpolate_points(points, labels, n_interp=n_interp)
    mt_img = _create_mt_image(points, labels, shape=shape)
    return mt_img


def _dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def _interpolate_points(points, labels, n_interp=10):
    new_points = []
    new_labels = []
    for lb in np.unique(labels):
        ind = np.where(labels == lb)
        coords = points[ind]
        if len(coords) > 1:
            coords_interp = np.concatenate([np.linspace(coords[i], coords[i + 1], n_interp, endpoint=False)
                                            for i in range(len(coords) - 1)], axis=0)
        else:
            coords_interp = coords
        new_points.append(coords_interp),
        new_labels = new_labels + [lb] * len(coords_interp)
    return np.concatenate(new_points, axis=0), np.array(new_labels)


def _create_mt_image(points, labels, shape):
    img = np.zeros(shape)
    if len(points) > 0:
        for lb in np.unique(labels):
            ind = np.where(labels == lb)
            coords = np.int_(points[ind]).transpose()
            img[tuple(coords)] = lb
    return img
