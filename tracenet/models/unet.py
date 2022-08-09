from monai.networks.layers import Norm
from monai.networks.nets import UNet


def get_unet(n_channels, num_res_units=1, spatial_dims=2, in_channels=1, out_channels=2):
    net = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=n_channels,
        strides=(2,) * (len(n_channels) - 1),
        num_res_units=num_res_units,
        norm=Norm.BATCH,
    )
    return net
