import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


class PointLoss(torch.nn.Module):

    def __init__(self, dist_push_weight=1., intensity_weight=1., mindist=0.05, maxval=255):
        super().__init__()
        self.dist_push_weight = dist_push_weight
        self.intensity_weight = intensity_weight
        self.mindist = mindist
        self.maxval = maxval

    def forward(self, trace, img):
        dist_push_loss = torch.stack([dist_push(tr, self.mindist) for tr in trace]).mean()
        imgf = gaussian_blur(img, kernel_size=[11]*2)
        int_loss = torch.stack([intensity_loss(im, tr, self.maxval) for im, tr in zip(imgf, trace)]).mean()
        return dist_push_loss * self.dist_push_weight + int_loss * self.intensity_weight


def diff_index(x, idx):
    x = F.pad(x.unsqueeze(0).unsqueeze(0), pad=(0, 2, 0, 2), mode='reflect').squeeze(0).squeeze(0)
    i0 = (idx[0] + 0.5).floor().detach()
    i1 = i0 + 1
    j0 = (idx[1] + 0.5).floor().detach()
    j1 = j0 + 1

    # index along first axis
    y0_ax0 = torch.stack([x.index_select(0, i0[i].long()).index_select(1, j0[i].long().detach())
                          for i in range(i0.shape[0])]).ravel()
    y1_ax0 = torch.stack([x.index_select(0, i1[i].long()).index_select(1, j0[i].long().detach())
                          for i in range(i1.shape[0])]).ravel()
    out_ax0 = (i1 - idx[0]) * y0_ax0 + (idx[0] - i0) * y1_ax0

    # index along second axis
    y0_ax1 = torch.stack([x.index_select(1, j0[i].long()).index_select(0, i0[i].long().detach())
                          for i in range(j0.shape[0])]).ravel()
    y1_ax1 = torch.stack([x.index_select(1, j1[i].long()).index_select(0, i0[i].long().detach())
                          for i in range(j1.shape[0])]).ravel()
    out_ax1 = (j1 - idx[1]) * y0_ax1 + (idx[1] - j0) * y1_ax1
    return (out_ax0 + out_ax1) / 2


def intensity_loss(img, trace, maxval=255):
    intensity = diff_index(img, (trace * (torch.tensor(img.shape).to(img.device).unsqueeze(0))).transpose(0, 1))
    return (maxval - intensity).mean()


def dist_push(trace, mindist=0.05):
    ind = np.array([(i, j) for i in range(len(trace)) for j in range(i + 1, len(trace))]).transpose()
    dist = torch.cdist(trace, trace)[tuple(ind)]
    return torch.clamp(mindist - dist, min=0).mean()
