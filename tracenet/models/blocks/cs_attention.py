"""
Adapted from here: https://github.com/iMED-Lab/CS-Net
"""
import torch
import torch.nn as nn


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels,
                               kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1,
                                 keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights
        return out


class SpatialAttentionBlock3d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8,
                               kernel_size=(1, 3, 1),
                               padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8,
                             kernel_size=(3, 1, 1),
                             padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8,
                               kernel_size=(1, 1, 3),
                               padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels,
                               kernel_size=(1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):

    def __init__(self):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1,
                                 keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):

    def __init__(self, in_channels, spacial_dims=2):
        super(AffinityAttention, self).__init__()
        if spacial_dims == 2:
            self.sab = SpatialAttentionBlock(in_channels)
            self.cab = ChannelAttentionBlock()
        elif spacial_dims == 3:
            self.sab = SpatialAttentionBlock3d(in_channels)
            self.cab = ChannelAttentionBlock3d()
        else:
            raise NotImplementedError(rf"{spacial_dims} is not a valid value for model dimension, must be 2 or 3.")

    def forward(self, x):
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out
