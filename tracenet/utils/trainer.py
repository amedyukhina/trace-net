import argparse
import datetime
import json
import os

import torch
import wandb
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.color import label2rgb

from .loader import get_loaders
from ..losses.contrastive import ContrastiveLoss
from ..models import get_model
from ..utils.plot import pca_project

DEFAULT_CONFIG = dict(
    backbone='monai_unet',
    epochs=20,
    batch_size=2,
    lr=0.0001,
    n_classes=1,
    mean_std=(0, 1),
    weight_decay=0.0005,
    factor=0.1,
    patience=10,
    model_path='model',
    log_wandb=False,
    log_tensorboard=True,
    wandb_project='Test',
    data_dir='data',
    train_dir='train',
    val_dir='val',
    img_dir='img',
    gt_dir='gt',
    maxsize=512,
    n_points=2,
    n_channels=(16, 32, 64, 128),
    num_res_units=2,
    spatial_dims=2,
    spoco=False,
    spoco_momentum=0.999,
    out_channels=16,
    instance=False,
    delta_var=0.5,
    delta_dist=3.,
    kernel_threshold=0.9,
    include_background=False,
    wandb_api_key_file='path_to_my_wandb_api_key_file'
)


class Trainer:
    def __init__(self, **kwargs):
        config = DEFAULT_CONFIG
        config.update(kwargs)
        self.config = argparse.Namespace(**config)

        # set up logging with tensorboard and wandb
        self.log_wandb = True if self.config.wandb_api_key_file is not None and \
                                 os.path.exists(self.config.wandb_api_key_file) else False
        self._init_project()
        self.best_model_name = 'best_model.pth'
        self.last_model_name = 'last_model.pth'
        self.tbwriter = SummaryWriter(log_dir=os.path.join(self.config.model_path, 'logs')) \
            if self.config.log_tensorboard else None

        # set data loaders and the model
        self.train_dl, self.val_dl = get_loaders(**config)
        self.net = get_model(self.config)

        # set loss function, validation metric, and forward pass depending on the model type
        if self.config.instance:
            dice_loss = DiceLoss(include_background=self.config.include_background)
            self.loss_function = ContrastiveLoss(self.config.delta_var,
                                                 self.config.delta_dist,
                                                 self.config.kernel_threshold,
                                                 instance_loss=dice_loss)
        else:
            self.loss_function = DiceLoss(include_background=self.config.include_background,
                                          to_onehot_y=True, softmax=True)

        self.metric = DiceMetric(include_background=self.config.include_background,
                                 reduction="mean")

        # send the model and loss to cuda if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.to(self.device)
        self.loss_function.to(self.device)

        # set optimizer and learning rate scheduler
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.config.lr,
                                           weight_decay=self.config.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.factor,
            patience=self.config.patience
        )

        # set loss weight coefficients
        self.weight_dict = {}

    def __del__(self):
        if self.log_wandb:
            wandb.finish()
        self.save_model()

    def _init_project(self):
        if self.log_wandb:
            with open(self.config.wandb_api_key_file) as f:
                key = f.read()
            os.environ['WANDB_API_KEY'] = key
        else:
            os.environ['WANDB_MODE'] = 'offline'

        wandb.init(project=self.config.wandb_project, config=vars(self.config))

        # Update model path
        self.config.model_path = os.path.join(self.config.model_path, self._get_model_name())

        # Save training parameters
        os.makedirs(self.config.model_path, exist_ok=True)
        with open(os.path.join(self.config.model_path, 'config.json'), 'w') as f:
            params = vars(self.config)
            params['data_dir'] = str(params['data_dir'])
            json.dump(params, f, indent=4)

    def _get_model_name(self):
        if self.log_wandb:
            model_name = wandb.run.name
        else:
            model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return model_name

    def log_scalar_tb(self, key, value, step):
        if self.tbwriter is not None:
            self.tbwriter.add_scalar(key, value, step)

    def train(self):
        best_loss = 10 ** 10
        for epoch in range(self.config.epochs):
            # training pass
            train_loss = self.train_epoch()

            # log training losses
            print(f"epoch {epoch + 1} training loss: {train_loss:.4f}")
            wandb.log({'average training loss': train_loss,
                       'epoch': epoch + 1,
                       'lr': self.optimizer.param_groups[0]['lr']})
            self.log_scalar_tb('average training loss', train_loss, epoch + 1)
            self.log_scalar_tb('learning rate', self.optimizer.param_groups[0]['lr'], epoch + 1)

            # validation pass
            val_loss = self.validate_epoch()

            # log validation losses and metrics
            print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
            wandb.log({'validation loss': val_loss})
            self.log_scalar_tb('validation loss', val_loss, epoch + 1)
            if self.metric is not None:
                val_metric = self.metric.aggregate().item()
                self.metric.reset()
                wandb.log({self.metric.__class__.__name__: val_metric})
                self.log_scalar_tb(self.metric.__class__.__name__, val_metric, epoch + 1)

            # update learning rate
            self.lr_scheduler.step(val_loss)

            # save the model state dict
            self.save_model()  # save last weights (default mode=0)
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(mode=1)  # save best weights (mode=1)

            # log images to tensorboard
            self.log_images(epoch + 1)

    def forward_pass(self, batch):
        imgs1, imgs2, targets = batch
        if self.config.spoco:
            outputs = self.net(imgs1.to(self.device),
                               imgs2.to(self.device))
        else:
            outputs = self.net(imgs1.to(self.device))
        return outputs, targets

    def calculate_losses(self, outputs, targets, metric=None):
        if self.config.instance:
            losses = self.loss_function(outputs, targets['labeled_mask'].to(self.device), metric)
        else:
            losses = self.loss_function(outputs, targets['mask'].unsqueeze(1).to(self.device))
            if metric is not None:
                metric(outputs.argmax(1).unsqueeze(1),
                       targets['mask'].unsqueeze(1).to(self.device))
        return losses

    def train_epoch(self):
        self.net.train()
        self.loss_function.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(self.train_dl):
            step += 1
            self.optimizer.zero_grad()
            outputs, targets = self.forward_pass(batch)
            losses = self.calculate_losses(outputs, targets)
            losses.backward()
            self.optimizer.step()
            epoch_loss += losses.item()

            wandb.log({'training loss': losses.item()})
            self.log_scalar_tb('training loss', losses.item(), step)
        epoch_loss /= step
        return epoch_loss

    def validate_epoch(self):
        self.net.eval()
        self.loss_function.eval()
        epoch_loss = 0
        step = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dl):
                step += 1
                outputs, targets = self.forward_pass(batch)
                losses = self.calculate_losses(outputs, targets, self.metric)
                epoch_loss += losses.item()
            epoch_loss /= step
        return epoch_loss

    def save_model(self, mode=0):
        name = self.best_model_name if mode else self.last_model_name
        fn_out = os.path.join(self.config.model_path, name)
        os.makedirs(self.config.model_path, exist_ok=True)
        torch.save(self.net.state_dict(), fn_out)
        print(rf"Saved model to: {fn_out}")

    def _postproc_outputs_targets(self, _, outputs, targets):
        if self.config.instance:
            return pca_project(outputs[0].cpu().numpy()), \
                   label2rgb(targets['labeled_mask'][0].numpy(), bg_label=0)
        else:
            return outputs[0].argmax(0).unsqueeze(-1), targets['mask'][0].unsqueeze(-1)

    def log_images(self, iteration):
        if self.tbwriter is not None:
            batch = next(iter(self.val_dl))
            with torch.no_grad():
                outputs, targets = self.forward_pass(batch)
            self.tbwriter.add_image('input', _normalize(batch[0][0]), iteration, dataformats='CHW')
            output, target = self._postproc_outputs_targets(batch[0], outputs, targets)
            self.tbwriter.add_image('output', output, iteration, dataformats='HWC')
            self.tbwriter.add_image('target', target, iteration, dataformats='HWC')


def _normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img
