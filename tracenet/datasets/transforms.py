import albumentations as A
import numpy as np
import torch
import torch.utils.data
from albumentations.pytorch.transforms import ToTensorV2

KEYPOINT_PARAMS = A.KeypointParams(format='xy',
                                   label_fields=['point_labels'],
                                   remove_invisible=False,
                                   angle_in_degrees=True)


def collate_fn(batch):
    imgs, targets = tuple(zip(*batch))
    imgs = torch.stack(imgs)
    return imgs, targets


def get_train_transform_spatial():
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(p=1, border_mode=0, value=0, limit=10),
        A.Transpose(p=0.5),
        ToTensorV2(p=1.0)
    ], keypoint_params=KEYPOINT_PARAMS)


def get_train_transform_intensity():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(p=1.0)
    ], keypoint_params=KEYPOINT_PARAMS)


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], keypoint_params=KEYPOINT_PARAMS)


def apply_transform(transforms, target, image):
    ndim = target['boxes'].shape[-1]
    if ndim != 4:
        boxes = target['boxes'].reshape(-1, 2)
    key = 'keypoints'
    boxes = target['boxes']
    sample = {'image': image, 'labels': target['labels'], key: boxes}
    if 'point_labels' in target:
        sample['point_labels'] = target['point_labels']
    sample2 = transforms(**sample)
    while len(sample2[key]) == 0:
        sample2 = transforms(**sample)

    image = sample2['image'].float()
    target['boxes'] = torch.tensor(np.array(sample2[key])).float().reshape(-1, ndim)
    for col in ['labels', 'iscrowd']:
        if col in target:
            target[col] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)
    return target, image


def norm_pad_to_gray(image, maxsize=None):
    image = image / np.max(image)
    if len(image.shape) > 2:
        image = np.max(image, axis=-1)
    image = image.astype(np.float32)
    if maxsize is None:
        maxsize = np.max(image.shape)
    else:
        maxsize = max(maxsize, np.max(image.shape))
    image = np.pad(image, [(0, maxsize - image.shape[0]), (0, maxsize - image.shape[1])])
    return image
