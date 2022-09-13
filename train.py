import argparse

from tracenet.utils.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Directory with the data (training, validation, test)', required=True)
    parser.add_argument('-s', '--maxsize', type=int, default=1024,
                        help='Maximum image size')
    parser.add_argument('-m', '--model', type=str,
                        help='Model type ("unet", "csnet", or "tracenet")', required=True)
    parser.add_argument('-n', '--n-points', type=int, default=2,
                        help='Number of points in the trace')
    parser.add_argument('-t', '--train-dir', type=str, default='train',
                        help='Subdirectory within "data-dir" to use for training')
    parser.add_argument('-v', '--val-dir', type=str, default='val',
                        help='Subdirectory within "data-dir" to use for validation')
    parser.add_argument('-im', '--img-dir', type=str, default='val',
                        help='Subdirectory for input image data')
    parser.add_argument('-gt', '--gt-dir', type=str, default='val',
                        help='Subdirectory for ground truth data')
    parser.add_argument('-mp', '--model-path', type=str,
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
    parser.add_argument('-c', '--n-channels', type=str, default="16,32,64,128",
                        help='Number of channels in each UNet layers, separated by ","')
    parser.add_argument('-oc', '--out-channels', type=int, default=16,
                        help='Number of output channels (embedding space) for instance segmentation')
    parser.add_argument('-r', '--num-res-units', type=int, default=1,
                        help='Number of residual units in each block of Unet and CSNet')
    parser.add_argument('-wp', '--wandb-project', type=str, default='',
                        help='wandb project name')
    parser.add_argument('-log', '--log-tensorboard', action='store_true')
    parser.add_argument('-dsegm', '--dsegm', action='store_true',
                        help='If True, use the segmentation data loader')
    parser.add_argument('-instance', '--instance', action='store_true',
                        help='If True, do instance segmentation, otherwise semantic')
    parser.add_argument('-spoco', '--spoco', action='store_true')
    parser.add_argument('-inclbg', '--include-background', action='store_true')
    parser.add_argument('-wapi', '--wandb-api-key-file', type=str, default=None,
                        help='Path to the wandb api key file')

    config = parser.parse_args()
    config.n_channels = [int(i) for i in config.n_channels.split(',')]
    if config.dsegm:
        config.segm_only = True
    else:
        config.segm_only = False
    config = vars(config)

    print('\nThe following are the parameters that will be used:')
    print(config)
    print('\n')
    trainer = Trainer(**config)
    trainer.train()
