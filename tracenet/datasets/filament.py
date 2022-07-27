import numpy as np
import pandas as pd
import torch
import torch.utils.data
from skimage import io

from .transforms import apply_transform, normalize
from ..utils import normalize_points


class FilamentDetection(torch.utils.data.Dataset):
    def __init__(self, image_files, ann_files, transforms=None,
                 col_id='id', maxsize=None, n_points=10, cols=None):
        self.transforms = transforms
        self.ann_files = ann_files
        self.image_files = image_files
        self.col_id = col_id
        self.maxsize = maxsize
        self.n_points = n_points
        self.cols = cols if cols is not None else ['x', 'y']

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = normalize(io.imread(image_id), maxsize=self.maxsize)

        df = pd.read_csv(ann_id)
        boxes = []
        for s in df[self.col_id].unique():
            cur_df = df[df[self.col_id] == s].reset_index(drop=True)
            if len(cur_df[self.cols].values.ravel()) != 2 * self.n_points:
                cur_df = make_points_equally_spaced(cur_df, self.cols, n_points=self.n_points)
            coords = cur_df[self.cols].values.ravel()
            boxes.append(coords)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

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

        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)
        target['boxes'] = normalize_points(target['boxes'], image.shape[-2:])
        target['area'] = torch.ones((target['boxes'].shape[0],), dtype=torch.float32)
        return image, target

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


def _dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def _interpolate_points(df, cols, n_interp=10):
    new_df = pd.DataFrame()
    coords = df[cols].values
    for i in range(len(coords) - 1):
        new_coords = np.array([np.linspace(coords[i, j], coords[i + 1, j], n_interp, endpoint=False)
                               for j in range(len(coords[i]))]).transpose()
        cur_df = pd.DataFrame(new_coords, columns=cols)
        new_df = pd.concat([new_df, cur_df], ignore_index=True)
    return new_df
