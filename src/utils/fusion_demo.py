"""
CICI Fusion Architecture Demo
===============================
Late-fusion model combining:
  - Imaging branch: frozen 3D U-Net encoder -> global average pooling -> FC
  - Tabular branch: XGBoost risk score (scalar)
  -> Concatenation -> Linear -> Sigmoid -> CICI probability

This is an architecture stub for the SBMT 2026 poster methods panel.
It demonstrates the full pipeline concept with placeholder forward passes.
No training is performed — real training requires paired imaging + clinical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenUNetEncoder(nn.Module):
    """3D U-Net encoder (V2 architecture, frozen weights).

    Extracts a feature vector from a multi-channel brain MRI volume
    by running the encoder path and bottleneck, then applying global
    average pooling and a fully-connected projection.

    Architecture mirrors UNet3D_V2 from model_3d_v2.py:
      features = [64, 128, 256, 320], bottleneck = 640
    """

    def __init__(self, in_channels=2, features=(64, 128, 256, 320), fc_dim=128):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        ch = in_channels
        for f in features:
            self.encoders.append(nn.Sequential(
                nn.Conv3d(ch, f, 3, padding=1, bias=False),
                nn.InstanceNorm3d(f),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(f, f, 3, padding=1, bias=False),
                nn.InstanceNorm3d(f),
                nn.LeakyReLU(0.01, inplace=True),
            ))
            ch = f

        bottleneck_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features[-1], bottleneck_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(bottleneck_ch, bottleneck_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(bottleneck_ch, fc_dim)

    def forward(self, x):
        for enc in self.encoders:
            x = self.pool(enc(x))
        x = self.bottleneck(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class CICIFusionModel(nn.Module):
    """Late-fusion CICI risk prediction.

    Concatenates an imaging feature vector with a tabular risk score,
    then passes through a linear classification head.

    Pipeline diagram (matches PLAN.md):
        MRI volume -> FrozenUNetEncoder -> [128-d vector]
                                                          ┐
                                                          ├─ concat -> Linear -> Sigmoid
                                                          ┘
        Clinical features -> XGBoost -> [1-d risk score]
    """

    def __init__(self, imaging_dim=128, tabular_dim=1, hidden_dim=64):
        super().__init__()
        self.imaging_encoder = FrozenUNetEncoder(fc_dim=imaging_dim)
        self.fusion = nn.Sequential(
            nn.Linear(imaging_dim + tabular_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mri_volume, tabular_risk_score):
        img_features = self.imaging_encoder(mri_volume)
        combined = torch.cat([img_features, tabular_risk_score], dim=1)
        return torch.sigmoid(self.fusion(combined))


def demo_forward_pass():
    """Run a placeholder forward pass to verify the architecture."""
    device = "cpu"
    model = CICIFusionModel().to(device)

    # Freeze imaging encoder (simulates pretrained weights)
    for param in model.imaging_encoder.parameters():
        param.requires_grad = False

    batch_size = 2
    mri = torch.randn(batch_size, 2, 128, 128, 128, device=device)
    risk_score = torch.tensor([[0.73], [0.31]], device=device)

    model.eval()
    with torch.no_grad():
        output = model(mri, risk_score)

    print("=" * 60)
    print("CICI Fusion Architecture — Demo Forward Pass")
    print("=" * 60)
    print(f"\nInput MRI shape:      {tuple(mri.shape)}")
    print(f"Input risk score:     {risk_score.squeeze().tolist()}")
    print(f"Output CICI prob:     {output.squeeze().tolist()}")
    print(f"\nImaging encoder params: "
          f"{sum(p.numel() for p in model.imaging_encoder.parameters()):,} "
          f"(frozen)")
    print(f"Fusion head params:    "
          f"{sum(p.numel() for p in model.fusion.parameters()):,} "
          f"(trainable)")
    print(f"Total params:          "
          f"{sum(p.numel() for p in model.parameters()):,}")

    print("\n--- Architecture Summary ---")
    print("Imaging branch:  MRI (2x128^3) -> 4-stage encoder -> GAP -> FC(128)")
    print("Tabular branch:  Clinical features -> XGBoost -> risk score (1)")
    print("Fusion:          concat(128+1) -> Linear(129->64) -> ReLU -> "
          "Dropout(0.3) -> Linear(64->1) -> Sigmoid")
    print("Output:          CICI probability in [0, 1]")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    demo_forward_pass()
