import albumentations as A
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

from .custom_transforms import GaussianNoise, GaussianBlur, RandomBrightnessContrast

KEYPOINT_PARAMS = A.KeypointParams(format='yx',
                                   label_fields=['point_labels'],
                                   remove_invisible=True,
                                   angle_in_degrees=True)


def collate_fn(batch):
    imgs1, imgs2, targets = tuple(zip(*batch))
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    target = dict()
    for key in ['mask', 'labeled_mask', 'padding']:
        target[key] = torch.stack([targets[i][key] for i in range(len(targets))])

    for key in ['keypoints', 'point_labels', 'trace', 'trace_class']:
        target[key] = [targets[i][key] for i in range(len(targets))]
    return imgs1, imgs2, target


def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(p=1, border_mode=0, value=0, limit=30),
        A.Transpose(p=0.5),
    ], keypoint_params=KEYPOINT_PARAMS)


def get_intensity_transform():
    return transforms.Compose([
        GaussianBlur([1.5, 2], 0.5),
        # GaussianNoise([0.02, 0.08], 0.5),
        # RandomBrightnessContrast(brightness=(-0.1, 0.1), contrast=(0.8, 1.2), probability=0.5),
    ])


def apply_transform(transforms, target, image):
    sample = {'image': image,
              'point_labels': target['point_labels'],
              'keypoints': target['keypoints']}
    sample2 = transforms(**sample)
    while len(sample2['keypoints']) == 0:
        sample2 = transforms(**sample)

    image = sample2['image']
    target['keypoints'] = np.array(sample2['keypoints'])
    target['point_labels'] = np.array(sample2['point_labels'])
    return target, image


def reshape_image_for_transformer(img, n):
    """
    Reshapes the image for a spatial transformer from size NxN to C x n x n.
    The input image size N must be a multiple of the new size n.

    Parameters
    ----------
    img : torch.tensor
        Input image
    n : int
        New image size

    Returns
    -------

    """
    s = int(img.shape[-1] / n)
    img = torch.moveaxis(img.reshape(img.shape[:-2] + (n, s, n, s)), -3, -2).reshape(img.shape[:-2] + (n, n, -1))
    img = torch.moveaxis(img, -1, -3)
    if len(img.shape) > 4:
        img = img.reshape((img.shape[0], -1) + img.shape[-2:])
    return img
