"""3D U-Net for volumetric brain tumor segmentation.

InstanceNorm3d + LeakyReLU throughout; feature widths kept small
([32, 64, 128, 256]) to fit comfortably on a 24 GB RTX 3090 with
(128, 128, 128) patches and batch size 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        features = list(features)

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        ch = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f))

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)
