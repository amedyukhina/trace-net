import numpy as np
import pandas as pd
import torch
import torch.utils.data
from skimage import io

from .transforms import apply_transform, normalize
from ..utils import normalize_points


class FilamentDetection(torch.utils.data.Dataset):
    def __init__(self, image_files, ann_files, transforms=None, col_id='id', maxsize=None):
        self.transforms = transforms
        self.ann_files = ann_files
        self.image_files = image_files
        self.col_id = col_id
        self.maxsize = maxsize

    def __getitem__(self, index: int):
        image_id = self.image_files[index]
        ann_id = self.ann_files[index]

        image = normalize(io.imread(image_id), maxsize=self.maxsize)

        df = pd.read_csv(ann_id)
        boxes = []
        for s in df[self.col_id].unique():
            coords = df[df[self.col_id] == s][['x', 'y']].values.ravel()
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
        return image, target

    def __len__(self) -> int:
        return len(self.image_files)
