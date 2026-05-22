"""Decoder head for Clay cocoa segmentation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ClayCocoaSegHead(nn.Module):
    def __init__(self, embed_dim: int = 384, out_size: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(self.head(x), size=self.out_size, mode="bilinear", align_corners=False)
