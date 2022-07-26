import albumentations as A
import numpy as np
import torch
import torch.utils.data
from albumentations.pytorch.transforms import ToTensorV2


def collate_fn(batch):
    imgs, targets = tuple(zip(*batch))
    imgs = torch.stack(imgs)
    return imgs, targets


def get_train_transform(**kwargs):
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(p=1, border_mode=0, value=0, limit=30),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(p=1.0)
    ], **kwargs)


def get_valid_transform(**kwargs):
    return A.Compose([
        ToTensorV2(p=1.0)
    ], **kwargs)


def apply_transform(transforms, target, image):
    ndim = target['boxes'].shape[-1]
    if ndim != 4:
        boxes = target['boxes'].reshape(-1, 2)
        key = 'keypoints'
    else:
        boxes = target['boxes']
        key = 'bboxes'
    sample = {'image': image, 'labels': target['labels'], key: boxes}
    if 'point_labels' in target:
        sample['point_labels'] = target['point_labels']
    sample2 = transforms(**sample)
    while len(sample2[key]) == 0:
        sample2 = transforms(**sample)

    image = sample2['image'].float()
    target['boxes'] = torch.tensor(np.array(sample2[key])).float().reshape(-1, ndim)
    return target, image


def normalize(image, maxsize=None):
    image = image / np.max(image)
    if len(image.shape) < 3:
        image = np.dstack([np.array(image)] * 3)
    image = image.astype(np.float32)
    if maxsize is None:
        maxsize = np.max(image.shape)
    else:
        maxsize = max(maxsize, np.max(image.shape))
    image = np.pad(image, [(0, maxsize - image.shape[0]), (0, maxsize - image.shape[1]), (0, 0)])
    return image
