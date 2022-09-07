from .csnet import get_csnet
from .detr import get_detr
from .unet import get_unet
from .spoco_unet import get_spoco_unet


def get_model(config):
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
            in_channels=3, out_channels=2
        )
    elif config.model.lower() == 'csnet':
        net = get_csnet(
            n_channels=config.n_channels,
            num_res_units=config.num_res_units,
            spatial_dims=config.spatial_dims,
            in_channels=3, out_channels=2
        )
    elif config.model.lower() == 'spoco_unet':
        net = get_spoco_unet(
            n_channels=config.n_channels,
            in_channels=3, out_channels=2
        )
    else:
        raise ValueError(rf"{config.model} is not a valid model; "
                         " valid models are: 'tracenet', 'unet', 'csnet', 'spoco_unet'")

    return net
