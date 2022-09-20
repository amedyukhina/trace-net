import torch.nn as nn
from monai.networks.nets import UNet

from tracenet.models.blocks.cs_attention import AffinityAttention


class CSNet(UNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        mod = nn.Sequential(
            self._get_down_layer(in_channels, out_channels, 1, False),
            AffinityAttention(out_channels)
        )
        return mod
