import numpy as np
import pandas as pd
import torch
import torch.utils.data
from skimage import io
from torchvision.transforms import ToTensor

from .transforms import apply_transform, norm_to_gray, pad_to_max, get_valid_transform
from .transforms_segm import get_valid_transform_segm
from ..utils.points import normalize_points, points_to_bounding_line, get_first_and_last_points


class FilamentDetection(torch.utils.data.Dataset):

    def __init__(self, image_files, ann_files, transforms=None, intensity_transforms=None,
                 instance_ratio=1, col_id='id', maxsize=None, n_points=10, cols=None):
        self.transforms = transforms if transforms is not None \
            else get_valid_transform()
        self.intensity_transforms = intensity_transforms
        self.instance_ratio = instance_ratio
        self.ann_files = ann_files
        self.image_files = image_files
        self.seeds = np.random.randint(0, np.iinfo('int32').max, len(image_files))
        self.col_id = col_id
        self.maxsize = maxsize
        self.n_points = n_points
        self.cols = cols if cols is not None else ['y', 'x']

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = pad_to_max(norm_to_gray(io.imread(image_id)), maxsize=self.maxsize)
        image = np.dstack([image] * 3)

        points, labels = df_to_points(pd.read_csv(ann_id), self.cols, self.col_id)
        target = dict(
            keypoints=points,
            image_id=torch.tensor([index]),
            point_labels=labels,
        )

        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)

        mask = torch.tensor(generate_labeled_mask(target['keypoints'].numpy(),
                                                  target['point_labels'].numpy(),
                                                  image.shape[-2:], n_interp=30),
                            dtype=torch.int64)
        target['boxes'] = points_to_bounding_line(
            get_first_and_last_points(
                normalize_points(target['keypoints'], image.shape[-2:]),
                target['point_labels']
            )
        )
        target['labels'] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)
        return image, image, target, mask, (mask > 0) * 1

    def __len__(self) -> int:
        return len(self.image_files)


class FilamentSegmentation(FilamentDetection):

    def __init__(self, image_files, ann_files, transforms=None, **kwargs):
        super(FilamentSegmentation, self).__init__(image_files, ann_files,
                                                   transforms=get_valid_transform_segm()
                                                   if transforms is None else transforms,
                                                   **kwargs)

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = norm_to_gray(io.imread(image_id))
        image = ToTensor()(np.dstack([image] * 3))

        mask = io.imread(ann_id)
        if self.instance_ratio < 1:
            mask = sample_instances_from_img(mask, self.instance_ratio, seed=self.seeds[index])
        mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)

        seed = np.random.randint(np.iinfo('int32').max)
        torch.manual_seed(seed)
        image = self.transforms(image)
        torch.manual_seed(seed)
        mask = self.transforms(mask).squeeze(0)

        if self.intensity_transforms is not None:
            image1 = self.intensity_transforms(image)
            image2 = self.intensity_transforms(image)
        else:
            image1 = image2 = image

        return image1, image2, dict(), mask, (mask > 0) * 1

    def __len__(self) -> int:
        return len(self.image_files)


def sample_instances_from_img(mask, instance_ratio, min_instances=2, seed=None):
    llist = np.unique(mask)[1:]
    n_objects = int(max(min_instances, round(instance_ratio * len(llist))))
    np.random.seed(seed)
    np.random.shuffle(llist)
    mask[np.in1d(mask.ravel(), llist[n_objects:]).reshape(mask.shape)] = 0
    return mask


# def make_points_equally_spaced(df, cols, n_points=10):
#     """
#     Interpolate the filament to a given number of points.
#     """
#     df = _interpolate_points(df.copy(), cols, n_interp=3)
#
#     # create empty image
#     img = np.zeros(np.array([int(df[c].max()) + 10 for c in cols]))
#
#     # calculate the distance profile
#     coords = df[cols].values
#     dist = np.array([0] + list(_dist(coords[:-1], coords[1:])))
#     dist = np.cumsum(dist)
#     dist_img = np.zeros_like(img)
#     dist_img[tuple(np.int_(coords.transpose()))] = dist
#
#     # new distance profile
#     n_dist = np.linspace(dist[0], dist[-1], n_points)
#
#     # new coordinates
#     n_coords = [coords[0]]
#     for i in range(1, len(n_dist)):
#         diff_img = np.abs(dist_img - n_dist[i])
#         n_coords.append(np.array(np.where(diff_img == np.min(diff_img))).transpose()[0])
#
#     n_df = pd.DataFrame(n_coords, columns=cols)
#     return n_df


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
