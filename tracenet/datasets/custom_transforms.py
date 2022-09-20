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
        np.random.seed()
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
        np.random.seed()
        if np.random.random() < self.probability:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            ks = int((sigma * 4 // 2) * 2 + 1)
            return F.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=sigma)
        else:
            return tensor


class RandomBrightnessContrast:
    def __init__(self, brightness=0.1, contrast=1., probability=1):
        brightness = np.ravel([brightness])
        contrast = np.ravel([contrast])
        if len(brightness) == 1:
            brightness = [0, brightness[0]]
        if len(contrast) == 1:
            contrast = [0, contrast[0]]
        self.brightness = brightness
        self.contrast = contrast
        self.probability = probability

    def __call__(self, tensor):
        np.random.seed()
        if np.random.random() < self.probability:
            br = np.random.uniform(self.brightness[0], self.brightness[1])
            c = np.random.uniform(self.contrast[0], self.contrast[1])
            return (tensor + br) * c
        else:
            return tensor
