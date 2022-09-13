import os

import pytest

from tracenet.utils.trainer import Trainer


def _assert_output(trainer):
    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))


def test_trainer(example_data_path, model_path, model_type):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, train_dir='', val_dir='',
                      model=model_type, batch_size=1, epochs=2)
    trainer.train()
    _assert_output(trainer)


def test_trainer_segm(example_segm_data_path, model_path, model_type_segm):
    trainer = Trainer(data_dir=example_segm_data_path, model_path=model_path,
                      train_dir='', val_dir='', segm_only=True,
                      model=model_type_segm, batch_size=1, epochs=2, maxsize=64)
    trainer.train()
    _assert_output(trainer)


@pytest.fixture(params=[2, 32, 64, 128])
def out_channels(request):
    return request.param


def test_instance_model(example_segm_data_path, model_path, model_type_segm, out_channels):
    trainer = Trainer(data_dir=example_segm_data_path, model_path=model_path,
                      train_dir='', val_dir='', segm_only=True, instance=True,
                      model=model_type_segm, batch_size=1, epochs=2, maxsize=64,
                      out_channels=out_channels)
    imgs = next(iter(trainer.train_dl))[0].to(trainer.device)
    trainer.net.to(trainer.device).eval()
    outputs = trainer.net(imgs)
    assert outputs.shape[-2:] == imgs.shape[-2:]
    assert outputs.shape[1] == trainer.config.out_channels


def test_trainer_instance_segm(example_segm_data_path, model_path, model_type_segm):
    trainer = Trainer(data_dir=example_segm_data_path, model_path=model_path,
                      train_dir='', val_dir='', segm_only=True, instance=True,
                      model=model_type_segm, batch_size=1, epochs=2, maxsize=64,
                      out_channels=64)
    trainer.train()
    _assert_output(trainer)
