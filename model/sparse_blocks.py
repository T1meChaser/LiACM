"""TorchSparse building blocks for LiACM.

English: compact sparse modules used for occupancy generation, coordinate
generation, residual sparse convolutions, and target/depth embedding.
"""

import torch
import torch.nn as nn
from torchsparse import SparseTensor
from torchsparse import nn as spnn


class ResNet(torch.nn.Module):
    """Residual sparse convolution block."""

    def __init__(self, channels, k=3):
        super().__init__()
        self.conv0 = spnn.Conv3d(channels, channels, k)
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out = self.relu(out + x)
        return out


class FOG(torch.nn.Module):
    """Fast occupancy converter."""

    def __init__(self):
        super(FOG, self).__init__()

        self.conv = spnn.Conv3d(1, 1, kernel_size=2, stride=2, bias=False)
        torch.nn.init.constant_(self.conv.kernel, 1.0)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.pos_multiplier = torch.tensor([[1, 2, 4]], device="cuda")

    def pos(self, coords):
        pos = (coords[:, 1:] % 2) * self.pos_multiplier
        pos = pos.sum(dim=-1, keepdim=True)
        pos = (2 ** pos).float()
        return pos

    def forward(self, x):
        x.feats = self.pos(x.coords)
        ds_x = self.conv(x)
        return ds_x


class FCG(torch.nn.Module):
    """Fast coordinate converter."""

    def __init__(self):
        super(FCG, self).__init__()

        self.expand_coords_base = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            device="cuda",
        )

        self.pos = torch.arange(0, 8, device="cuda").view(1, 8)

    def forward(self, x_C, x_O, x_F=None):
        expand_coords = self.expand_coords_base.repeat(x_C.shape[0], 1)
        x_C_repeat = x_C.repeat(1, 8).reshape(-1, 4)
        x_C_repeat[:, 1:] = x_C_repeat[:, 1:] * 2 + expand_coords
        mask = torch.div(
            x_O.repeat(1, 8) % (2 ** (self.pos + 1)),
            2 ** self.pos,
            rounding_mode="floor",
        ).reshape(-1)
        mask = mask == 1
        x_up_C = x_C_repeat[mask].int()
        if x_F is None:
            return x_up_C
        else:
            channels = x_F.shape[1]
            x_F = x_F.repeat(1, 8).reshape(-1, channels)
            x_up_F = x_F[mask]
            return x_up_C, x_up_F


class TargetEmbedding(torch.nn.Module):
    """Target child-position embedding."""

    def __init__(self, channels):
        super(TargetEmbedding, self).__init__()
        self.target_res_embedding = nn.Embedding(8, channels)

    def forward(self, x_up_F, x_up_C):
        coords_delta = x_up_C[:, 1:] % 2
        coords_idx = coords_delta[:, 0] + coords_delta[:, 1] * 2 + coords_delta[:, 2] * 4
        x_up_F = x_up_F + self.target_res_embedding(coords_idx.int())
        return x_up_F


class LightweightMultiScaleFusion2(nn.Module):
    """Lightweight sparse multi-scale fusion block used by LiACM."""

    def __init__(self, channels, k=3):
        super().__init__()

        self.conv_1x1 = spnn.Conv3d(channels, channels // 4, 1)
        self.conv_3x3 = spnn.Conv3d(channels, channels // 4, 3)
        self.conv_5x5 = spnn.Conv3d(channels, channels // 2, 5)

        self.fusion = spnn.Conv3d(channels, channels, 1)
        self.relu = spnn.ReLU(True)
        # All fusion tensors retain the input coordinates, order and stride.
        self.preserve_cache = True

    def forward(self, x):
        identity = x.feats

        feat_1 = self.conv_1x1(x).feats
        feat_3 = self.conv_3x3(x).feats
        feat_5 = self.conv_5x5(x).feats

        multi_scale = torch.cat([feat_1, feat_3, feat_5], dim=1)
        fusion_input = SparseTensor(coords=x.coords, feats=multi_scale,
                                    stride=x.stride, spatial_range=x.spatial_range)
        if self.preserve_cache:
            fusion_input._caches = x._caches
        fused = self.fusion(fusion_input)

        out = SparseTensor(coords=x.coords, feats=fused.feats + identity,
                           stride=x.stride, spatial_range=x.spatial_range)
        if self.preserve_cache:
            out._caches = fused._caches
        out = self.relu(out)
        return out
