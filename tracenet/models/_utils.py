from .csnet import get_csnet
from .detr import get_detr
from .spoco import SpocoNet
from .spoco_unet import get_spoco_unet
from .unet import get_unet


def _get_model(config):
    if config.instance:
        out_channels = config.out_channels
    else:
        out_channels = config.n_classes + 1
    if config.model.lower() == 'tracenet':
        net = get_detr(
            n_classes=config.n_classes,
            n_points=config.n_points,
            pretrained=True
        )
    elif config.model.lower() == 'unet':
        net = get_unet(
            n_channels=config.n_channels,
            num_res_units=config.num_res_units,
            spatial_dims=config.spatial_dims,
            in_channels=3, out_channels=out_channels
        )
    elif config.model.lower() == 'csnet':
        net = get_csnet(
            n_channels=config.n_channels,
            num_res_units=config.num_res_units,
            spatial_dims=config.spatial_dims,
            in_channels=3, out_channels=out_channels
        )
    elif config.model.lower() == 'spoco_unet':
        net = get_spoco_unet(
            n_channels=config.n_channels,
            in_channels=3, out_channels=out_channels
        )
    else:
        raise ValueError(rf"{config.model} is not a valid model; "
                         " valid models are: 'tracenet', 'unet', 'csnet', 'spoco_unet'")

    return net


def get_model(config):
    net = _get_model(config)
    if config.spoco:
        net2 = _get_model(config)
        return SpocoNet(net, net2, m=config.spoco_momentum)
    else:
        return net
