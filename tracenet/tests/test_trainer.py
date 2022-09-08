import os

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
