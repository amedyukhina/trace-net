import argparse
import json
import os
from pathlib import Path

import albumentations as A
import wandb
from torch.utils.data import DataLoader

from tracenet import get_train_transform, get_valid_transform, collate_fn
from tracenet.datasets import FilamentDetection
from tracenet.models.criterion import Criterion
from tracenet.models.detr import build_model
from tracenet.models.matcher import HungarianMatcher
from tracenet.utils import get_model_name
from tracenet.utils.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data-dir', type=str,
                        help='Directory with the data (training, validation, test)', required=True)
    parser.add_argument('-s', '--maxsize', type=int, default=1024,
                        help='Maximum image size')
    parser.add_argument('-n', '--n-points', type=int, default=10,
                        help='Number of points in the trace')
    parser.add_argument('-t', '--train-dir', type=str, default='train',
                        help='Subdirectory within "data-dir" to use for training')
    parser.add_argument('-v', '--val-dir', type=str, default='val',
                        help='Subdirectory within "data-dir" to use for validation')
    parser.add_argument('-m', '--model-path', type=str,
                        help='Directory for model checkpoints', default='model')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='Starting learning rate')
    parser.add_argument('-bl', '--bbox-loss-coef', type=float, default=5,
                        help='Weight for bbox loss')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0005,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('-f', '--factor', type=float, default=0.1,
                        help='Factor parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-p', '--patience', type=int, default=10,
                        help='Patience parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-pr', '--wandb-project', type=str, default='',
                        help='wandb project name')
    parser.add_argument('-log', '--log-progress', action='store_true')

    config = parser.parse_args()

    print('\nThe following are the parameters that will be used:')
    print(vars(config))
    print('\n')

    # Initialize wandb project
    if config.log_progress:
        with open('/home/amedyukh/.wandb_api_key') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key
    else:
        os.environ['WANDB_MODE'] = 'offline'

    wandb.init(project=config.wandb_project, config=vars(config))

    # Update model path
    config.model_path = os.path.join(config.model_path, get_model_name(config.log_progress))

    # Save training parameters
    os.makedirs(config.model_path, exist_ok=True)
    with open(os.path.join(config.model_path, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    # Setup data loaders
    path = Path(config.data_dir)
    ds = []
    for dset, transform in zip([config.train_dir, config.val_dir],
                               [get_train_transform, get_valid_transform]):
        files = os.listdir(path / dset / 'img')
        files.sort()
        ds.append(
            FilamentDetection(
                [path / dset / 'img' / fn for fn in files],
                [path / dset / 'gt' / fn.replace('.tif', '.csv') for fn in files],
                transforms=transform(keypoint_params=A.KeypointParams(format='xy',
                                                                      label_fields=['point_labels'],
                                                                      remove_invisible=False,
                                                                      angle_in_degrees=True)),
                maxsize=config.maxsize, n_points=config.n_points
            )
        )
    ds_train, ds_val = ds

    dl_train = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn,
                          batch_size=config.batch_size, num_workers=config.batch_size)
    dl_val = DataLoader(ds_val, shuffle=False, collate_fn=collate_fn,
                        batch_size=config.batch_size, num_workers=config.batch_size)

    # Setup model, loss, and metric
    model = build_model(n_classes=1, n_points=config.n_points, pretrained=True)
    loss_function = Criterion(1, HungarianMatcher(), losses=['labels', 'boxes', 'cardinality'])

    # train
    train(dl_train, dl_val, model, loss_function, config=config, log_tensorboard=True)
    wandb.finish()
