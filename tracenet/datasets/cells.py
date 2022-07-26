import pandas as pd
import torch
import torch.utils.data
from skimage import io

from .transforms import apply_transform, normalize
from ..utils import xyxy_to_cxcywh, normalize_points


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

        boxes = torch.as_tensor(records[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)

        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                               dtype=torch.float32)

        # there is only one class
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
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
        target['boxes'] = normalize_points(xyxy_to_cxcywh(target['boxes']), image.shape[-2:])
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]
