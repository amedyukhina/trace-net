import os

import pytest

from tracenet.utils.trainer import Trainer


def _assert_output(trainer):
    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))


def test_trainer(example_data_path, model_path, backbone):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, train_dir='', val_dir='',
                      backbone=backbone, batch_size=1, epochs=2)
    trainer.train()
    _assert_output(trainer)


def test_unetr(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, train_dir='', val_dir='',
                      backbone='unetr', batch_size=1, epochs=2)
    trainer.train()
    _assert_output(trainer)


@pytest.fixture(params=[2, 32, 64, 128])
def out_channels(request):
    return request.param


@pytest.fixture(params=[
    [8, 16, 32, 128, 256, 64],
    [8, 16, 32, 128],
    [8, 16, 32, 128, 256],
    [8, 16, 32, 64, 128, 256, 32]
])
def n_channels(request):
    return request.param


def test_instance_model(example_data_path, model_path, out_channels):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      train_dir='', val_dir='', instance=True,
                      backbone='monai_unet', batch_size=1, epochs=2, out_channels=out_channels)
    imgs = next(iter(trainer.train_dl))[0].to(trainer.device)
    trainer.net.to(trainer.device).eval()
    outputs = trainer.net(imgs)
    assert outputs.shape[-2:] == imgs.shape[-2:]
    assert outputs.shape[1] == trainer.config.out_channels


def test_trainer_instance(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      train_dir='', val_dir='', instance=True,
                      backbone='monai_unet', batch_size=1, epochs=2, out_channels=64)
    trainer.train()
    _assert_output(trainer)


def test_tracenet_loss(example_data_path, model_path, n_channels, backbone):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, n_channels=n_channels,
                      backbone=backbone,
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    imgs, _, targets = next(iter(trainer.train_dl))
    imgs = imgs.to(trainer.device)
    for key in ['trace', 'trace_class']:
        targets[key] = [t.to(trainer.device) for t in targets[key]]
    trainer.net.to(trainer.device).eval()
    outputs = trainer.net(imgs)
    outputs['pred_traces'][0][:targets['trace'][0].shape[0]] = targets['trace'][0]
    outputs['pred_logits'][0][:, 0] = 100
    outputs['pred_logits'][0][:, 1] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 0] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 1] = 100
    loss_dict = trainer.loss_function_trace(outputs, targets)
    for key in loss_dict.keys():
        assert loss_dict[key].item() == 0


def test_tracenet_training(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, n_channels=[8, 16, 32, 64, 128, 32],
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    trainer.train()
    _assert_output(trainer)
