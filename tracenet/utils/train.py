import os

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from ..utils import get_device, save_model
from ..utils.plot import plot_results


def __log_images(writer, samples, targets, outputs, iteration):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    writer.add_image('input', samples[0].numpy().transpose(1, 2, 0) * 255,
                     iteration, dataformats='HWC')
    writer.add_image('output', plot_results(samples[0],
                                            outputs['pred_boxes'][0, keep], probas[keep],
                                            return_image=True),
                     iteration, dataformats='HWC')
    writer.add_image('target', plot_results(samples[0], targets['boxes'][0], return_image=True),
                     iteration, dataformats='HWC')


def __forward_pass(samples, targets, model, device, loss_function, weight_dict):
    samples = samples.to(device)
    targets = {k: v.to(device) for k, v in targets.items() if isinstance(v, torch.Tensor)}
    outputs = model(samples)
    loss_dict = loss_function(outputs, targets)
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return losses


def train(train_dl, val_dl, model, loss_function, config, log_tensorboard=False):
    device = get_device()
    model.to(device)
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
            losses = __forward_pass(samples, targets, model, device, loss_function, weight_dict)
            loss_value = losses.item()
            losses.backward()
            optimizer.step()
            epoch_loss += loss_value
            wandb.log({'training loss': loss_value})
            if log_tensorboard:
                tbwriter.add_scalar('training loss', loss_value, step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} training loss: {epoch_loss:.4f}")
        wandb.log({'average training loss': epoch_loss,
                   'epoch': epoch + 1,
                   'lr': optimizer.param_groups[0]['lr']})

        if log_tensorboard:
            tbwriter.add_scalar('average training loss', epoch_loss, epoch + 1)
            tbwriter.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch + 1)

        model.eval()
        val_loss = 0
        step = 0
        with torch.no_grad():
            for samples, targets in val_dl:
                step += 1
                val_losses = __forward_pass(samples, targets, model, device, loss_function, weight_dict)
                loss_value = val_losses.item()
                epoch_loss += loss_value

            val_loss /= step

        print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
        wandb.log({'validation loss': val_loss})

        if log_tensorboard:
            tbwriter.add_scalar('validation loss', epoch_loss, epoch + 1)
            samples, targets = next(iter(val_dl))
            with torch.no_grad():
                outputs = model(samples.to(device)).cpu()
            __log_images(tbwriter, samples, targets, outputs, epoch + 1)

        lr_scheduler.step(epoch_loss)

        if val_loss < best_metric:
            best_metric = val_loss
            save_model(model, config.model_path)
