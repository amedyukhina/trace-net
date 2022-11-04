import pytest
import torch
import os

from tracenet.losses.indexing import PointLoss
from tracenet.models.transformer import Transformer
from tracenet.utils.trainer import Trainer


@pytest.fixture(params=[512, 1024, 256])
def hidden_dim(request):
    return request.param


def _assert_output(trainer):
    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))


def test_transformer_model(hidden_dim):
    x = torch.rand(4, 1, 16, 16)
    net = Transformer(hidden_dim, n_points=1).cuda()
    output = net(x.cuda()).cpu()
    loss_fn = PointLoss(maxval=x.max())
    loss = loss_fn(output.cuda(), x.squeeze(1).cuda())
    assert loss.item() > 0


def test_tracenet_training(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      train_dir='', val_dir='', batch_size=1, epochs=2,
                      backbone='transformer')
    trainer.train()
    _assert_output(trainer)