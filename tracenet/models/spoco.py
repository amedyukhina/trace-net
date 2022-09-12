"""
from https://github.com/kreshuklab/spoco
"""
import torch
import torch.nn as nn


class SpocoNet(nn.Module):
    def __init__(self, net_f, net_g, m=0.999, init_equal=True):
        super(SpocoNet, self).__init__()

        self.net_f = net_f
        self.net_g = net_g
        self.m = m

        if init_equal:
            # initialize g weights to be equal to f weights
            for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
                param_g.data.copy_(param_f.data)  # initialize
                param_g.requires_grad = False  # freeze g parameters

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the g
        """
        for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
            param_g.data = param_g.data * self.m + param_f.data * (1. - self.m)

    def forward(self, im_f, im_g):
        # compute f-embeddings
        emb_f = self.net_f(im_f)

        # compute g-embeddings
        with torch.no_grad():  # no gradient to g-embeddings
            self._momentum_update()  # momentum update of g
            emb_g = self.net_g(im_g)

        return emb_f, emb_g
