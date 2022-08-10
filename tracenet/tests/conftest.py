import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def random_imgsize():
    return np.random.randint(50, 200, 2)


@pytest.fixture(scope='module')
def random_bboxes(random_imgsize):
    n = np.random.randint(5, 20)
    boxes = np.array([np.random.randint(0, s - 5, n) for s in random_imgsize])
    boxes2 = np.array([np.random.randint(b, s)
                       for i, s in enumerate(random_imgsize) for b in boxes[i]]).reshape(2, -1)
    return torch.as_tensor(np.concatenate([boxes, boxes2], axis=0).transpose())


@pytest.fixture(scope='module')
def random_points(random_imgsize):
    n = np.random.randint(5, 20)
    size = list(random_imgsize) * np.random.randint(5, 20)
    return torch.as_tensor(np.array([np.random.randint(0, s, n) for s in size]).transpose()).to(float)


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


@pytest.fixture(scope='module', params=['unet', 'csnet', 'tracenet'])
def model_type(request):
    return request.param


@pytest.fixture(scope='module', params=np.random.randint(0, 100, 10))
def random_coord(random_imgsize):
    return torch.as_tensor(np.array([np.random.randint(0, s, 2)
                                     for s in random_imgsize]).transpose()).to(float).ravel()
