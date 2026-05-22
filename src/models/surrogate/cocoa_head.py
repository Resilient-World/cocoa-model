"""
Linear and LoRA probe heads on top of frozen Galileo embeddings for cocoa parcel
segmentation and yield-loss regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearSegHead(nn.Module):
    """Per-token linear classifier for cocoa-vs-not segmentation."""

    def __init__(self, embed_dim: int, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, N, C]
        return self.fc(tokens)


class YieldLossHead(nn.Module):
    """Per-parcel MLP regressor for yield-loss percentage."""

    def __init__(self, embed_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, parcel_embed: torch.Tensor) -> torch.Tensor:
        # [B, D] -> [B]
        return self.mlp(parcel_embed).squeeze(-1)
