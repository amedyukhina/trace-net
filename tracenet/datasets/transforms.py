import albumentations as A
import numpy as np
import torch
import torch.utils.data
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

KEYPOINT_PARAMS = A.KeypointParams(format='yx',
                                   label_fields=['point_labels'],
                                   remove_invisible=True)


def collate_fn(batch):
    imgs1, imgs2, targets, labels, masks = tuple(zip(*batch))
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    return imgs1, imgs2, targets, labels, masks


def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(p=1, border_mode=0, value=0, limit=10),
        A.Transpose(p=0.5),
        ToTensorV2(p=1.0)
    ], keypoint_params=KEYPOINT_PARAMS)


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], keypoint_params=KEYPOINT_PARAMS)


def get_intensity_transform():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.1)
    ])


def apply_transform(transforms, target, image):
    sample = {'image': image,
              'point_labels': target['point_labels'],
              'keypoints': target['keypoints']}
    sample2 = transforms(**sample)
    while len(sample2['keypoints']) == 0:
        sample2 = transforms(**sample)

    image = sample2['image'].float()
    target['keypoints'] = torch.tensor(np.array(sample2['keypoints']), dtype=torch.float)
    target['point_labels'] = torch.tensor(sample2['point_labels'], dtype=torch.int64)
    return target, image


def norm_to_gray(image):
    image = image / np.max(image)
    if len(image.shape) > 2:
        image = np.max(image, axis=-1)
    image = image.astype(np.float32)
    return image


def pad_to_max(image, maxsize=None):
    if maxsize is None:
        maxsize = np.max(image.shape)
    else:
        maxsize = max(maxsize, np.max(image.shape))
    image = np.pad(image, [(0, maxsize - image.shape[0]), (0, maxsize - image.shape[1])])
    return image
