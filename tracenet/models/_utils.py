from monai.networks.layers import Norm
from monai.networks.nets import UNet, AttentionUnet

from .backbones.csnet import CSNet
from .backbones.spoco_unet import UNet2D as SpocoBackbone
from .spoco import SpocoNet
from .tracenet import TraceNet
from .transformer import Transformer


def get_backbone(config):
    if config.instance:
        out_channels = config.out_channels
    else:
        out_channels = config.n_classes + 1
    if config.backbone.lower() == 'monai_unet':
        net = UNet(
            spatial_dims=config.spatial_dims,
            in_channels=3,
            out_channels=out_channels,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            num_res_units=config.num_res_units,
            norm=Norm.BATCH,
            dropout=config.dropout
        )
        feature_layer = 'model' + '.1.submodule' * (len(config.n_channels) - 1)
    elif config.backbone.lower() == 'attention_unet':
        net = AttentionUnet(
            spatial_dims=config.spatial_dims,
            in_channels=3,
            out_channels=out_channels,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            dropout=config.dropout
        )
        feature_layer = 'model' + '.1.submodule' * (len(config.n_channels) - 1)
    elif config.backbone.lower() == 'csnet':
        net = CSNet(
            spatial_dims=config.spatial_dims,
            in_channels=3,
            out_channels=out_channels,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            num_res_units=config.num_res_units,
            norm=Norm.BATCH,
            dropout=config.dropout
        )
        feature_layer = 'model' + '.1.submodule' * (len(config.n_channels) - 1)
    elif config.backbone.lower() == 'spoco_unet':
        net = SpocoBackbone(
            in_channels=3,
            out_channels=out_channels,
            f_maps=config.n_channels,
            layer_order="bcr"
        )
        feature_layer = rf'encoders.{len(config.n_channels) - 1}'
    else:
        raise ValueError(rf"{config.backbone} is not a valid backbone; "
                         " valid backbones are: 'monai_unet', 'csnet', 'spoco_unet'")

    return net, feature_layer


def get_model(config):
    if config.backbone.lower() == 'transformer':
        net = Transformer(hidden_dim=256, n_points=1)
        return net
    else:
        net, feature_layer = get_backbone(config)
        if config.tracing:
            net = TraceNet(backbone=net,
                           transformer_input_layer=feature_layer,
                           hidden_dim=config.n_channels[-1],
                           num_classes=config.n_classes,
                           n_points=config.n_points)
            return net
        elif config.spoco:
            net2 = get_backbone(config)
            return SpocoNet(net, net2, m=config.spoco_momentum)
        else:
            return net
