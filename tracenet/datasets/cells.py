import pandas as pd
import torch
import torch.utils.data
from skimage import io
import numpy as np

from .transforms import apply_transform, normalize
from ..utils import xyxy_to_cxcywh, normalize_points


def crop_out_of_shape(boxes, shape):
    boxes[:, 0] = np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
    boxes[:, 1] = np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
    boxes[:, 2] = np.where(boxes[:, 2] >= shape[1], shape[1] - 1, boxes[:, 2])
    boxes[:, 3] = np.where(boxes[:, 3] >= shape[0], shape[0] - 1, boxes[:, 3])
    return boxes


class CellDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, maxsize=None):
        self.transforms = transforms
        self.df = pd.read_csv(ann_file)
        self.image_ids = self.df['image_id'].unique()
        self.image_dir = img_folder
        self.maxsize = maxsize

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]

        image = normalize(io.imread(f'{self.image_dir}/{image_id}'), maxsize=self.maxsize)
        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        boxes = crop_out_of_shape(boxes, image.shape[:2])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                               dtype=torch.float32)
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = dict(
            boxes=boxes,
            labels=labels,
            image_id=torch.tensor([index]),
            area=area,
            iscrowd=iscrowd,
        )

        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)
        x2, x1, y2, y1 = (target['boxes'][:, 3], target['boxes'][:, 1], target['boxes'][:, 2], target['boxes'][:, 0])
        target['area'] = torch.as_tensor((x2 - x1) * (y2 - y1), dtype=torch.float32)
        target['boxes'] = normalize_points(xyxy_to_cxcywh(target['boxes']), image.shape[-2:])
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]
