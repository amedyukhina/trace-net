import argparse

from tracenet.utils.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Directory with the data (training, validation, test)', required=True)
    parser.add_argument('-s', '--maxsize', type=int, default=512,
                        help='Maximum image size')
    parser.add_argument('-t', '--train-dir', type=str, default='train',
                        help='Subdirectory within "data-dir" to use for training')
    parser.add_argument('-v', '--val-dir', type=str, default='val',
                        help='Subdirectory within "data-dir" to use for validation')
    parser.add_argument('-im', '--img-dir', type=str, default='val',
                        help='Subdirectory for input image data')
    parser.add_argument('-gt', '--gt-dir', type=str, default='val',
                        help='Subdirectory for ground truth data')
    parser.add_argument('-ms', '--mean-std', type=str, default="0,1",
                        help='Mean and standard deviation of the dataset, for normalization, separated by ","')
    parser.add_argument('-mp', '--model-path', type=str,
                        help='Directory for model checkpoints', default='model')
    parser.add_argument('-mpp', '--pretrained-model-path', type=str,
                        help='Pretrained model weights', default=None)
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-bs', '--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='Starting learning rate')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0005,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('-fr', '--factor', type=float, default=0.1,
                        help='Factor parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-pt', '--patience', type=int, default=10,
                        help='Patience parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-wp', '--wandb-project', type=str, default='',
                        help='wandb project name')
    parser.add_argument('-log', '--log-tensorboard', action='store_true')
    parser.add_argument('-wapi', '--wandb-api-key-file', type=str, default=None,
                        help='Path to the wandb api key file')
    parser.add_argument('-np', '--n-points', type=int, default=2,
                        help='Number of points in the trace')
    parser.add_argument('-wtt', '--weight-trace', type=float, default=1.,
                        help='Weight for the trace coordinates in the loss function')
    parser.add_argument('-wte', '--weight-ends', type=float, default=2.,
                        help='Weight for the trace end coordinates in the loss function')
    parser.add_argument('-wts', '--weight-spacing', type=float, default=0.5,
                        help='Weight for the trace spacing in the loss function')
    parser.add_argument('-sm', '--symmetric', action='store_true', help='calculate trace distance in a symmetric way')

    config = parser.parse_args()
    config.mean_std = tuple([float(i) for i in config.mean_std.split(',')])
    config = vars(config)

    print('\nThe following are the parameters that will be used:')
    print(config)
    print('\n')
    trainer = Trainer(**config)
    trainer.train()
