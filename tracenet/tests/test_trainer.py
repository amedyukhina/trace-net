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

#
# def test_trainer(example_data_path, model_path, n_points, symmetric):
#     trainer = Trainer(data_dir=example_data_path, model_path=model_path, symmetric=symmetric,
#                       train_dir='', val_dir='', batch_size=1, epochs=2, n_points=n_points, random_flip=True)
#     trainer.train()
#     _assert_output(trainer)


def test_trainer_bezier(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, symmetric=False, bezier=True,
                      train_dir='', val_dir='', batch_size=1, epochs=2, n_points=5, random_flip=True)
    trainer.train()
    _assert_output(trainer)


# def _get_params(net):
#     params = []
#     for module in [net.backbone, net.transformer, net.query_embed, net.input_proj, net.class_embed]:
#         params = params + [param.cpu().detach().numpy() for param in module.parameters()]
#     return params
#
#
# def test_pretrained(example_data_path, model_path):
#     trainer = Trainer(data_dir=example_data_path, model_path=model_path + '/pretrained', symmetric=True,
#                       train_dir='', val_dir='', batch_size=1, epochs=2, n_points=3, random_flip=True)
#     trainer.train()
#     params1 = _get_params(trainer.net.detr)
#
#     trainer2 = Trainer(data_dir=example_data_path, model_path=model_path + '/second', symmetric=True,
#                        train_dir='', val_dir='', batch_size=1, epochs=2, n_points=5, random_flip=True,
#                        pretrained_model_path=trainer.config.model_path + '/last_model.pth')
#     params2 = _get_params(trainer2.net.detr)
#
#     for param1, param2 in zip(params1, params2):
#         assert (param1 == param2).all()
