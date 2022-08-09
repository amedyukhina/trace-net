import torch.nn as nn
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from .blocks import AffinityAttention


class CSNet(UNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        mod = nn.Sequential(
            self._get_down_layer(in_channels, out_channels, 1, False),
            AffinityAttention(out_channels)
        )
        return mod


def get_csnet(n_channels, num_res_units=1, spatial_dims=2, in_channels=1, out_channels=2):
    net = CSNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=n_channels,
        strides=(2,) * (len(n_channels) - 1),
        num_res_units=num_res_units,
        norm=Norm.BATCH,
    )
    return net
