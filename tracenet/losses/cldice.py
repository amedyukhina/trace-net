"""
Adapted from https://github.com/jocpae/clDice

Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

from .contrastive import expand_as_one_hot


# Soft morphology and skeletonization

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# Soft clDice loss

class SoftClDice(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftClDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_true = expand_as_one_hot(y_true, y_pred.shape[1])
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class SoftDiceClDice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, **kwargs):
        super(SoftDiceClDice, self).__init__()
        self.alpha = alpha
        self.cldice = SoftClDice(iter_, 1)
        self.dice = DiceLoss(**kwargs)

    def forward(self, y_pred, y_true):
        cl_dice = self.cldice(y_pred, y_true)
        dice = self.dice(y_pred, y_true.unsqueeze(1))
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
