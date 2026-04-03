"""3D U-Net V2 for volumetric brain tumor segmentation.

Changes from V1:
  - Larger feature widths: [64, 128, 256, 320] (vs [32, 64, 128, 256])
  - Deep supervision: auxiliary segmentation heads at intermediate decoder
    stages, weighted in the training loss to improve gradient flow.

VRAM note: ~4x more parameters than V1. Use batch_size=2 with AMP on
a 24 GB GPU. Run a 1-epoch sanity check before committing to a full run.
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


class UNet3D_V2(nn.Module):
    def __init__(self, in_channels=2, out_channels=1,
                 features=(64, 128, 256, 320), deep_supervision=True):
        super().__init__()
        features = list(features)
        self.deep_supervision = deep_supervision

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        ch = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(ch, f))
            ch = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        upconv_in = [features[-1] * 2] + rev[:-1]
        for ic, f in zip(upconv_in, rev):
            self.upconvs.append(nn.ConvTranspose3d(ic, f, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f))

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList(
                nn.Conv3d(f, out_channels, kernel_size=1) for f in rev[:-1]
            )

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        decoder_outs = []
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
            decoder_outs.append(x)

        main_out = self.head(x)

        if self.training and self.deep_supervision:
            aux = [head(feat) for head, feat in zip(self.ds_heads, decoder_outs[:-1])]
            return [main_out] + aux

        return main_out
