import albumentations as A
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms

from .custom_transforms import GaussianNoise, GaussianBlur

KEYPOINT_PARAMS = A.KeypointParams(format='yx',
                                   label_fields=['point_labels'],
                                   remove_invisible=True)

TARGET_KEYS = ['keypoints', 'point_labels', 'mask', 'labeled_mask', 'padding']


def collate_fn(batch):
    imgs1, imgs2, targets = tuple(zip(*batch))
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    target = dict()
    for key in TARGET_KEYS:
        target[key] = torch.stack([targets[i][key] for i in range(len(targets))])
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
        GaussianNoise([0.02, 0.08], 0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.1),
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
    target['point_labels'] = sample2['point_labels']
    return target, image
