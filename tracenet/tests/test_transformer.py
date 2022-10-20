import os

from tracenet.losses.indexing import PointLoss
from tracenet.models.transformer import Transformer
from tracenet.utils.loader import get_loaders
from tracenet.utils.trainer import Trainer


def _assert_output(trainer):
    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))


def test_transformer_model(example_data_path):
    loader = get_loaders(example_data_path, train_dir='', val_dir='', batch_size=1, mean_std=(3, 0.4))[0]
    _, _, target = next(iter(loader))
    mask = target['mask']
    net = Transformer(1024, n_points=1).cuda()
    output = net(mask.cuda()).cpu()
    loss_fn = PointLoss(maxval=mask.max())
    loss_fn(output.cuda(), mask.float().cuda())


def test_trainsformer_training(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, train_dir='', val_dir='',
                      backbone='transformer', batch_size=1, epochs=2)
    trainer.train()
    _assert_output(trainer)
