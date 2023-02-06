import argparse
import copy
import datetime
import json
import os

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loader import get_loaders
from ..losses.criterion import Criterion
from ..models.detr import DETR
from ..utils.plot import normalize, plot_traces
from ..utils.points import bezier_curve_from_control_points

DEFAULT_CONFIG = dict(
    epochs=20,
    batch_size=2,
    lr=0.0001,
    n_classes=1,
    mean_std=(0, 1),
    percentiles=(0, 100),
    weight_decay=0.0005,
    factor=0.1,
    patience=10,
    gamma=None,
    model_path='model',
    pretrained_model_path=None,
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
    symmetric=True,
    weight_trace=1,
    weight_spacing=0.5,
    weight_ends=2,
    weight_straightness=0.05,
    lim_strt=5,
    random_flip=False,
    bezier=False,
    non_pretrained=False,
    wandb_api_key_file='path_to_my_wandb_api_key_file',
    seed=None
)


class Trainer:
    def __init__(self, **kwargs):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(kwargs)
        self.config = argparse.Namespace(**config)

        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        # set loss weight coefficients
        self.weight_dict = {'loss_class': 1,
                            'loss_trace_distance': self.config.weight_trace,
                            # 'loss_point_spacing': self.config.weight_spacing,
                            'loss_end_coords': self.config.weight_ends,
                            'loss_straightness': self.config.weight_straightness
                            }

        # set up logging with tensorboard and wandb
        self.log_wandb = True if self.config.wandb_api_key_file is not None and \
                                 os.path.exists(self.config.wandb_api_key_file) else False
        self._init_project()
        self.best_model_name = 'best_model.pth'
        self.last_model_name = 'last_model.pth'
        self.tbwriter = SummaryWriter(log_dir=os.path.join(self.config.model_path, 'logs')) \
            if self.config.log_tensorboard else None

        # set data loaders and the model
        # set data loaders and the model
        self.train_dl, self.val_dl = get_loaders(**config)
        self.net = DETR(n_points=self.config.n_points if self.config.bezier is False else 4,
                        n_classes=self.config.n_classes,
                        pretrained=not self.config.non_pretrained,
                        pretrained_model_path=self.config.pretrained_model_path,
                        bezier=self.config.bezier)

        # set loss function, validation metric, and forward pass depending on the model type
        self.loss_function = Criterion(self.config.n_classes,
                                       losses=self.weight_dict.keys(),
                                       symmetric=self.config.symmetric,
                                       bezier=self.config.bezier,
                                       lim_strt=self.config.lim_strt)

        # send the model and loss to cuda if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.to(self.device)
        self.loss_function.to(self.device)

        # set optimizer and learning rate scheduler
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                           lr=self.config.lr,
                                           weight_decay=self.config.weight_decay)

        if self.config.gamma is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.gamma
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.factor,
                patience=self.config.patience
            )

        self.trained = False

    def __del__(self):
        if self.log_wandb:
            wandb.finish()
        if self.trained:
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
        self.trained = True
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
            val_loss, val_loss_dicts = self.validate_epoch()

            # log validation losses and metrics
            print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
            wandb.log({'validation loss': val_loss})
            self.log_scalar_tb('validation loss', val_loss, epoch + 1)
            if val_loss_dicts[0] is not None:
                for key in val_loss_dicts[0].keys():
                    loss_val = np.mean([val_loss_dicts[i][key].item() for i in range(len(val_loss_dicts))])
                    wandb.log({rf'val {key}': loss_val})
                    self.log_scalar_tb(rf'val {key}', loss_val, epoch + 1)

            # update learning rate
            if self.config.gamma is not None:
                self.lr_scheduler.step()
            else:
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
        outputs = self.net(imgs1.to(self.device))
        return outputs, targets

    def calculate_losses(self, outputs, targets):
        for key in ['trace', 'trace_class']:
            targets[key] = [t.to(self.device) for t in targets[key]]
        loss_dict = self.loss_function(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k]
                     for k in loss_dict.keys() if k in self.weight_dict)
        return losses, loss_dict

    def train_epoch(self):
        self.net.train()
        self.loss_function.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(self.train_dl):
            step += 1
            self.optimizer.zero_grad()
            outputs, targets = self.forward_pass(batch)
            losses, _ = self.calculate_losses(outputs, targets)
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
        loss_dicts = []
        with torch.no_grad():
            for batch in tqdm(self.val_dl):
                step += 1
                outputs, targets = self.forward_pass(batch)
                losses, loss_dict = self.calculate_losses(outputs, targets)
                loss_dicts.append(loss_dict)
                epoch_loss += losses.item()
            epoch_loss /= step
        return epoch_loss, loss_dicts

    def save_model(self, mode=0):
        name = self.best_model_name if mode else self.last_model_name
        fn_out = os.path.join(self.config.model_path, name)
        os.makedirs(self.config.model_path, exist_ok=True)
        torch.save(self.net.state_dict(), fn_out)
        print(rf"Saved model to: {fn_out}")

    def _postproc_traces(self, imgs, outputs, targets):
        probas = outputs['pred_logits'].softmax(-1)[0, :, 1:]
        keep = probas.max(-1).values > 0.7
        pred_traces = outputs['pred_traces'][0, keep].cpu()
        if self.config.bezier:
            n = 20
            assert pred_traces.shape[-1] == 8
            pred_traces = bezier_curve_from_control_points(pred_traces.reshape(-1, 4, 2), n).reshape(-1, n * 2)
        return plot_traces(imgs[0][0].cpu(), pred_traces,
                           return_image=True, n_points=pred_traces.shape[-1] // 2), \
               plot_traces(imgs[0][0].cpu(), targets['trace'][0].cpu(),
                           return_image=True, n_points=self.config.n_points)

    def log_images(self, iteration):
        if self.tbwriter is not None:
            batch = next(iter(self.val_dl))
            with torch.no_grad():
                outputs, targets = self.forward_pass(batch)

            self.tbwriter.add_image('input', normalize(batch[0][0]), iteration, dataformats='CHW')
            output, target = self._postproc_traces(batch[0], outputs, targets)
            self.tbwriter.add_image('output_tracing', output, iteration, dataformats='HWC')
            self.tbwriter.add_image('target_tracing', target, iteration, dataformats='HWC')
