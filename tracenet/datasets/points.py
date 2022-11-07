import numpy as np
import torch
import torch.utils.data


class Points(torch.utils.data.Dataset):
    """
    Dataset with points.
    """

    def __init__(self, image_files, ann_files, maxsize=16, n_points=10, length=100, **_):
        self.maxsize = maxsize
        self.n_points = n_points
        self.length = length

    def __getitem__(self, index: int):
        mask = torch.zeros(self.maxsize, self.maxsize)

        points = np.random.rand(self.n_points, 2)
        labels = np.arange(self.n_points) + 1
        target = dict(
            keypoints=points,
            point_labels=labels,
        )
        mask[tuple(np.int_(points * self.maxsize).transpose())] = 1.
        mask = torch.tensor(mask)

        target['mask'] = mask.float()
        target['labeled_mask'] = mask.float()
        target['keypoints'] = torch.tensor(target['keypoints'], dtype=torch.float64)
        target['point_labels'] = torch.tensor(target['point_labels'], dtype=torch.int64)
        target['padding'] = torch.zeros(2, dtype=torch.float64)

        target['trace'] = target['keypoints']
        target['trace_class'] = torch.ones((target['trace'].shape[0],), dtype=torch.int64)

        return mask.unsqueeze(0), mask.unsqueeze(0), target

    def __len__(self) -> int:
        return self.length
