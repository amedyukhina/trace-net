import torch

from .blocks import MLP


def get_detr(n_classes=1, n_points=10, pretrained=True, state_dict_path=None):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=pretrained)
    hdim = model.transformer.d_model
    model.class_embed = torch.nn.Linear(hdim, n_classes + 1)
    if n_points != 2:
        model.bbox_embed = MLP(hdim, hdim, n_points * 2, 3)

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    return model
