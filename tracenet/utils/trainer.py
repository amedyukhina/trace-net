import os

import numpy as np
import torch
import wandb
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

from .loader import get_loaders
from ..losses import Criterion
from ..models import get_unet, get_detr, get_csnet
from ..utils import get_device, save_model
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
        elif config.model.lower() == 'unet':
            self.net = get_unet(
                n_channels=config.n_channels,
                num_res_units=config.num_res_units,
                spatial_dims=config.spatial_dims,
                in_channels=3, out_channels=2
            )
            self._set_dice_loss_and_metric()
        elif config.model.lower() == 'csnet':
            self.net = get_csnet(
                n_channels=config.n_channels,
                num_res_units=config.num_res_units,
                spatial_dims=config.spatial_dims,
                in_channels=3, out_channels=2
            )
            self._set_dice_loss_and_metric()
        else:
            raise ValueError(rf"{config.model} is not a valid model; "
                             " valid models are: 'tracenet', 'unet', 'csnet'")
        self.device = get_device()

    def _set_dice_loss_and_metric(self):
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")


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


def __forward_pass(samples, targets, model, device, loss_function, weight_dict):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
    outputs = model(samples)
    loss_dict = loss_function(outputs, targets)
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return losses, loss_dict


def train(train_dl, val_dl, model, loss_function, config, log_tensorboard=False):
    device = get_device()
    model.to(device)
    loss_function.to(device)
    best_metric = 10 ** 10
    param_dict = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dict, lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=config.factor,
                                                              patience=config.patience)
    tbwriter = None
    if log_tensorboard:
        tbwriter = SummaryWriter(log_dir=os.path.join(config.model_path, 'logs'))

    weight_dict = {'loss_ce': 1, 'loss_bbox': config.bbox_loss_coef}

    for epoch in range(config.epochs):
        model.train()
        loss_function.train()
        epoch_loss = 0
        step = 0
        for samples, targets in train_dl:
            step += 1
            optimizer.zero_grad()
            losses, loss_dict = __forward_pass(samples, targets, model, device, loss_function, weight_dict)
            loss_value = losses.item()
            losses.backward()
            optimizer.step()
            epoch_loss += loss_value
            wandb.log({'training loss': loss_value})
            wandb.log({k: loss_dict[k] for k in loss_dict.keys()})
            if log_tensorboard:
                tbwriter.add_scalar('training loss', loss_value, step)
                for k in loss_dict.keys():
                    tbwriter.add_scalar(k, loss_dict[k], step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} training loss: {epoch_loss:.4f}")
        wandb.log({'average training loss': epoch_loss,
                   'epoch': epoch + 1,
                   'lr': optimizer.param_groups[0]['lr']})

        if log_tensorboard:
            tbwriter.add_scalar('average training loss', epoch_loss, epoch + 1)
            tbwriter.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch + 1)

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
