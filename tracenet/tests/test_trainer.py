import os

import pytest

from tracenet.utils.trainer import Trainer


def _assert_output(trainer):
    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))


@pytest.fixture(params=[2, 3, 5, 10])
def n_points(request):
    return request.param


@pytest.fixture(params=[False, True])
def symmetric(request):
    return request.param


def test_trainer(example_data_path, model_path, n_points, symmetric):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, symmetric=symmetric,
                      train_dir='', val_dir='', batch_size=1, epochs=2, n_points=n_points, random_flip=True)
    trainer.train()
    _assert_output(trainer)
