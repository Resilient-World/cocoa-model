"""U-Net-style decoder for OlmoEarth patch features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OlmoEarthCocoaSegHead(nn.Module):
    def __init__(self, embed_dim: int = 512, out_size: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.out_size = out_size
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.decode(features)
        return F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
