import numpy as np
import torch
import torchvision.transforms.functional as F


class GaussianNoise:
    def __init__(self, scale=1., probability=1):
        scale = np.ravel([scale])
        if len(scale) == 1:
            scale = [0, scale[0]]
        self.scale = scale
        self.probability = probability

    def __call__(self, tensor):
        if np.random.random() < self.probability:
            std = np.random.uniform(self.scale[0], self.scale[1])
            return tensor + torch.randn(tensor.size()) * std
        else:
            return tensor


class GaussianBlur:
    def __init__(self, sigma=1., probability=1):
        sigma = np.ravel([sigma])
        if len(sigma) == 1:
            sigma = [0, sigma[0]]
        self.sigma = sigma
        self.probability = probability

    def __call__(self, tensor):
        if np.random.random() < self.probability:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            ks = int((sigma * 4 // 2) * 2 + 1)
            return F.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=sigma)
        else:
            return tensor


class Normalize:
    def __init__(self, min_value=0, max_value=1):
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max() * (self.max_value - self.min_value) + self.min_value
        return tensor
