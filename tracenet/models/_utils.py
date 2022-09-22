from monai.networks.layers import Norm

from .backbones.csnet import CSNet
from .backbones.monai_unet import monai_unet
from .backbones.spoco_unet import UNet2D as SpocoBackbone
from .detr import detr
from .spoco import SpocoNet


def get_backbone(config):
    if config.instance:
        out_channels = config.out_channels
    else:
        out_channels = config.n_classes + 1
    if config.backbone.lower() == 'monai_unet':
        net = monai_unet(
            n_channels=config.n_channels,
            num_res_units=config.num_res_units,
            spatial_dims=config.spatial_dims,
            in_channels=3, out_channels=out_channels
        )
    elif config.backbone.lower() == 'csnet':
        net = CSNet(
            spatial_dims=config.spatial_dims,
            in_channels=3,
            out_channels=out_channels,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            num_res_units=config.num_res_units,
            norm=Norm.BATCH,
        )
    elif config.backbone.lower() == 'spoco_unet':
        net = SpocoBackbone(
            in_channels=3,
            out_channels=out_channels,
            f_maps=config.n_channels,
            layer_order="bcr"
        )
    else:
        raise ValueError(rf"{config.backbone} is not a valid backbone; "
                         " valid backbones are: 'monai_unet', 'csnet', 'spoco_unet'")

    return net


def get_model(config):
    if config.tracing:
        net = detr(
            n_classes=config.n_classes,
            n_points=2,
            # n_points=config.n_points,
            pretrained=True
        )
    else:
        net = get_backbone(config)
    if config.spoco:
        net2 = get_backbone(config)
        return SpocoNet(net, net2, m=config.spoco_momentum)
    else:
        return net
