import os

from tracenet.utils.trainer import Trainer


def test_trainer(example_data_path, model_path, model_type):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path, train_dir='', val_dir='',
                      model=model_type, batch_size=1, epochs=2)

    assert os.path.exists(os.path.join(trainer.config.model_path, 'config.json'))
    trainer.train()
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.best_model_name))
    assert os.path.exists(os.path.join(trainer.config.model_path, trainer.last_model_name))
