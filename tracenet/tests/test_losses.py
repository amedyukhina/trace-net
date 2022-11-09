import numpy as np
import pytest
import torch

from tracenet.losses.indexing import intensity_loss, dist_push
from tracenet.utils.trainer import Trainer


@pytest.fixture(params=[5, 10, 100])
def trace(request):
    return torch.tensor(np.array([np.linspace(0, 1, request.param, endpoint=False),
                                  np.linspace(0, 1, request.param, endpoint=False)]).transpose()), request.param


def test_dist_push(trace):
    trace, npoints = trace
    dist = dist_push(trace, mindist=1. / npoints / 2)
    assert dist == 0
    dist = dist_push(trace, mindist=1. / npoints * 2)
    assert dist > 0


def test_diff_ind(trace):
    img = torch.ones([50, 50]) * 255
    assert intensity_loss(img, trace[0], maxval=255) == 0
    assert intensity_loss(img, trace[0], maxval=500) >= 0


def test_tracenet_loss(example_data_path, model_path):
    trainer = Trainer(data_dir=example_data_path, model_path=model_path,
                      train_dir='', val_dir='', batch_size=1, epochs=2, tracing=True, n_points=2)
    imgs, _, targets = next(iter(trainer.train_dl))
    imgs = imgs.to(trainer.device)
    for key in ['trace', 'trace_class']:
        targets[key] = [t.to(trainer.device) for t in targets[key]]
    trainer.net.to(trainer.device).eval()
    outputs = trainer.net(imgs)
    loss_dict = trainer.loss_function(outputs, targets)
    for key in loss_dict.keys():
        assert loss_dict[key].item() >= 0
    outputs['pred_traces'][0][:targets['trace'][0].shape[0]] = targets['trace'][0]
    outputs['pred_logits'][0][:, 0] = 100
    outputs['pred_logits'][0][:, 1] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 0] = -100
    outputs['pred_logits'][0][:targets['trace'][0].shape[0], 1] = 100
    loss_dict = trainer.loss_function(outputs, targets)
    for key in loss_dict.keys():
        assert loss_dict[key].item() == 0
