import pytest
import torch

from tracenet.datasets.transforms import reshape_image_for_transformer


@pytest.fixture(params=[([256, 256], 16, 16*16),
                        ([2, 3, 16, 16], 4, 4*4*3),
                        ([3, 16, 16], 4, 4*4)])
def test_image(request):
    shape, n, nch = request.param
    img = torch.ones(shape)
    return img, n, nch


def test_reshpae(test_image):
    img, n, nch = test_image
    img_reshaped = reshape_image_for_transformer(img, n)
    assert img_reshaped.shape[-3] == nch
    assert img_reshaped.shape[-1] * n == img.shape[-1]
