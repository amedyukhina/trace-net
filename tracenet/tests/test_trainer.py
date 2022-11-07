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


@pytest.fixture(params=[2, 32, 64, 128])
def out_channels(request):
    return request.param


@pytest.fixture(params=[False, True])
def b_line(request):
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
    loss_dict = trainer.loss_function_trace(outputs, targets)
    for key in loss_dict.keys():
        assert loss_dict[key].item() >= 0
    outputs['pred_traces'][0][:targets['trace'][0].shape[0]] = targets['trace'][0]
    outputs['pred_logits'][0][:, 0] = 100
    outputs['pred_logits'][0][:, 1] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 0] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 1] = 100
    loss_dict = trainer.loss_function_trace(outputs, targets)
    for key in loss_dict.keys():
        assert loss_dict[key].item() == 0


def test_tracenet_training(example_data_path, model_path, b_line):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, n_channels=[8, 16, 32, 64, 128, 32],
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2, b_line=b_line)
    trainer.train()
    _assert_output(trainer)


def test_tracenet_pretraining(example_data_path, model_path):
    # pretrain
    trainer = Trainer(data_dir=example_data_path, model_path=model_path + '/backbone',
                      n_channels=[8, 16, 32, 64, 128, 32], backbone='unetr',
                      train_dir='', val_dir='', batch_size=1, epochs=5, tracing=False, n_points=2)
    trainer.train()
    loss_fn = trainer.loss_function

    # evaluate
    imgs, _, targets = next(iter(trainer.train_dl))
    imgs = imgs.to(trainer.device)

    # without pretraining
    trainer = Trainer(data_dir=example_data_path, model_path=model_path + '/tracing',
                      n_channels=[8, 16, 32, 64, 128, 32], backbone='unetr',
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    trainer.net.to(trainer.device).eval()
    outputs1 = trainer.net(imgs)

    # with pretraining
    trainer = Trainer(data_dir=example_data_path, model_path=model_path + '/tracing',
                      pretrained_model_path=trainer.config.model_path + '/best_model.pth',
                      n_channels=[8, 16, 32, 64, 128, 32], backbone='unetr',
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    trainer.net.to(trainer.device).eval()
    outputs2 = trainer.net(imgs)
    loss1 = loss_fn(outputs1['backbone_out'], targets['mask'].to(trainer.device))
    loss2 = loss_fn(outputs2['backbone_out'], targets['mask'].to(trainer.device))
    assert loss2.item() < loss1.item()


def test_weight_freeze(example_data_path, model_path):
    # no weight freeze
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      freeze_backbone=False,
                      n_channels=[8, 16, 32, 64, 128, 32], backbone='unetr',
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    assert next(trainer.net.backbone.parameters()).requires_grad is True
    params1 = [param.cpu().detach().numpy() for param in trainer.net.backbone.parameters()]
    trainer.train()
    params2 = [param.cpu().detach().numpy() for param in trainer.net.backbone.parameters()]

    for param1, param2 in zip(params1, params2):
        assert (param1 != param2).any()

    # with weight freeze
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      freeze_backbone=True,
                      n_channels=[8, 16, 32, 64, 128, 32], backbone='unetr',
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    assert next(trainer.net.backbone.parameters()).requires_grad is False
    params1 = [param.cpu().detach().numpy() for param in trainer.net.backbone.parameters()]
    trainer.train()
    params2 = [param.cpu().detach().numpy() for param in trainer.net.backbone.parameters()]

    for param1, param2 in zip(params1, params2):
        assert (param1 == param2).all()
