import os

import numpy as np
import torch
import wandb
import json
import datetime
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

from .loader import get_loaders
from ..losses import Criterion
from ..models import get_unet, get_detr, get_csnet
from ..utils.plot import plot_results


class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_dl, self.val_dl = get_loaders(**vars(config))

        if config.model.lower() == 'tracenet':
            self.net = get_detr(
                n_classes=config.n_classes,
                n_points=config.n_points,
                pretrained=True
            )
            self.loss_function = Criterion(config.n_classes)
            self.metric = None
            self.forward_loss = self.forward_loss_tracenet
        elif config.model.lower() in ['unet', 'csnet']:
            get_model = get_unet if config.model.lower() == 'unet' else get_csnet
            self.net = get_model(
                n_channels=config.n_channels,
                num_res_units=config.num_res_units,
                spatial_dims=config.spatial_dims,
                in_channels=3, out_channels=2
            )
            self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
            self.metric = DiceMetric(include_background=False, reduction="mean")
            self.forward_loss = self.forward_loss_unet
        else:
            raise ValueError(rf"{config.model} is not a valid model; "
                             " valid models are: 'tracenet', 'unet', 'csnet'")

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.to(self.device)
        self.loss_function.to(self.device)

        self.optimizer = torch.optim.AdamW(
            params=[{"params": [p for p in self.net.parameters()
                                if p.requires_grad]}],
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.factor,
            patience=config.patience
        )

        self.weight_dict = {'loss_ce': 1, 'loss_bbox': config.bbox_loss_coef}

        self.tbwriter = SummaryWriter(log_dir=os.path.join(config.model_path, 'logs')) \
            if config.log_tensorboard else None

        self.log_wandb = True if config.api_key_file is not None and os.path.exists(config.api_key_file) else False
        self._init_project()

    def _init_project(self):
        if self.log_wandb:
            with open(self.config.api_key_file) as f:
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
            json.dump(vars(self.config), f, indent=4)

    def _get_model_name(self):
        if self.log_wandb:
            model_name = wandb.run.name
        else:
            model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return model_name

    def log_tb(self, key, value, step):
        if self.tbwriter is not None:
            self.tbwriter.add_scalar(key, value, step)

    def train(self):
        best_metric = 10 ** 10
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            print(f"epoch {epoch + 1} training loss: {train_loss:.4f}")
            wandb.log({'average training loss': train_loss,
                       'epoch': epoch + 1,
                       'lr': self.optimizer.param_groups[0]['lr']})

            self.log_tb('average training loss', train_loss, epoch + 1)
            self.log_tb('learning rate', self.optimizer.param_groups[0]['lr'], epoch + 1)

            val_loss = self.validate_epoch()
            print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
            wandb.log({'validation loss': val_loss})
            self.log_tb('validation loss', val_loss, epoch + 1)

            self.lr_scheduler.step(val_loss)

    def forward_loss_tracenet(self, batch, step):
        imgs, targets, _, _, = batch
        targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
        outputs = self.net(imgs.to(self.device))
        loss_dict = self.loss_function(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        if step is not None:
            wandb.log({k: loss_dict[k] for k in loss_dict.keys()})
            for k in loss_dict.keys():
                self.log_tb(k, loss_dict[k], step)
        return losses

    def forward_loss_unet(self, batch, _):
        imgs, _, _, masks = batch
        outputs = self.net(imgs.to(self.device))
        losses = self.loss_function(outputs, masks)
        return losses

    def train_epoch(self):
        self.net.train()
        self.loss_function.train()
        epoch_loss = 0
        step = 0
        for batch in self.train_dl:
            step += 1
            self.optimizer.zero_grad()
            losses = self.forward_loss(batch, step)
            losses.backward()
            self.optimizer.step()
            epoch_loss += losses.item()

            wandb.log({'training loss': losses.item()})
            self.log_tb('training loss', losses.item(), step)
        epoch_loss /= step
        return epoch_loss

    def validate_epoch(self):
        self.net.eval()
        self.loss_function.eval()
        epoch_loss = 0
        step = 0
        with torch.no_grad():
            for batch in self.val_dl:
                step += 1
                losses = self.forward_loss(batch, None)
                epoch_loss += losses.item()
            epoch_loss /= step
        return epoch_loss


    def save_model(self, model, model_path, model_name='best_model.pth'):
        fn_out = os.path.join(model_path, model_name)
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), fn_out)
        print(rf"Saved new best model to: {fn_out}")


def __normalize(img):
    img = img - np.min(img)
    return (img * 255. / np.max(img)).astype(np.uint8)


def __log_images(writer, samples, targets, outputs, iteration):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    writer.add_image('input', __normalize(samples[0].numpy().transpose(1, 2, 0)),
                     iteration, dataformats='HWC')
    writer.add_image('output', plot_results(samples[0].cpu(),
                                            outputs['pred_boxes'][0, keep].cpu(), probas[keep].cpu(),
                                            return_image=True),
                     iteration, dataformats='HWC')
    writer.add_image('target', plot_results(samples[0].cpu(), targets[0]['boxes'].cpu(), return_image=True),
                     iteration, dataformats='HWC')



def train(train_dl, val_dl, model, loss_function, config, log_tensorboard=False):
    for epoch in range(config.epochs):

        model.eval()
        loss_function.eval()
        val_loss = 0
        step = 0
        with torch.no_grad():
            for samples, targets in val_dl:
                step += 1
                val_losses = __forward_pass(samples, targets, model, device, loss_function, weight_dict)[0]
                loss_value = val_losses.item()
                val_loss += loss_value

            val_loss /= step

        print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
        wandb.log({'validation loss': val_loss})

        if log_tensorboard:
            tbwriter.add_scalar('validation loss', val_loss, epoch + 1)
            samples, targets = next(iter(val_dl))
            with torch.no_grad():
                outputs = model(samples.to(device))
            __log_images(tbwriter, samples, targets, outputs, epoch + 1)

        lr_scheduler.step(val_loss)

        if val_loss < best_metric:
            best_metric = val_loss
            save_model(model, config.model_path)
