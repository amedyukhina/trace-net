import pytest
import torch

from tracenet.losses.indexing import PointLoss
from tracenet.models.transformer import Transformer


@pytest.fixture(params=[512, 1024, 256])
def hidden_dim(request):
    return request.param


def test_transformer_model(hidden_dim):
    x = torch.rand(4, 1, 16, 16)
    net = Transformer(hidden_dim, n_points=1).cuda()
    output = net(x.cuda()).cpu()
    loss_fn = PointLoss(maxval=x.max())
    loss = loss_fn(output.cuda(), x.squeeze(1).cuda())
    assert loss.item() > 0
