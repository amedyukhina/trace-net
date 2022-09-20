import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from skimage import io

from tracenet.datasets.filament import generate_labeled_mask, df_to_points


@pytest.fixture(scope='module')
def random_imgsize():
    return np.random.randint(50, 200, 2)


@pytest.fixture(scope='module')
def random_points(random_imgsize):
    n = np.random.randint(5, 20)
    size = list(random_imgsize) * np.random.randint(5, 20)
    points = torch.as_tensor(np.array([np.random.randint(0, s, n) for s in size]).transpose()).to(float)
    npoints = []
    labels = []
    for i, point in enumerate(points):
        point = point.reshape(-1, 2)
        npoints.append(point)
        labels = labels + [i + 1] * len(point)
    return torch.concat(npoints), torch.tensor(labels)


@pytest.fixture(scope='module')
def example_data_path():
    cwd = os.path.dirname(os.path.abspath(__file__))
    return Path(os.path.abspath(os.path.join(cwd, '../../example_data')))


@pytest.fixture(scope='module')
def model_path():
    path = tempfile.mkdtemp()
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope='module', params=['monai_unet', 'csnet', 'spoco_unet'])
def backbone(request):
    return request.param


@pytest.fixture(scope='module', params=np.random.randint(0, 100, 10))
def random_coord(random_imgsize):
    return torch.as_tensor(np.array([np.random.randint(0, s, 2)
                                     for s in random_imgsize]).transpose()).to(float).ravel()
