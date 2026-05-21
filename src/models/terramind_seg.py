"""
TerraMind cocoa segmentation wrappers (standard + TiM).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data.utils import cocoa_batch_to_terramind_input
from models.terramind_backbone import TerraMindBackbone, load_terramind_backbone
from models.terramind_cocoa_head import TerraMindCocoaUPerNetHead
from models.terramind_tim import TerraMindTiM

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAMIND_CHECKPOINT = _REPO_ROOT / "models" / "terramind_cocoa_seg.pt"
DEFAULT_TERRAMIND_TIM_CHECKPOINT = _REPO_ROOT / "models" / "terramind_tim_cocoa_seg.pt"


class TerraMindCocoaSegmentation(nn.Module):
    """TerraMind encoder + UPerNet head for P(cocoa)."""

    def __init__(
        self,
        *,
        variant: str = "terramind_v1_base",
        freeze_backbone: bool = True,
        pretrained_backbone: bool = False,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = load_terramind_backbone(
            variant=variant, freeze=freeze_backbone, pretrained=pretrained_backbone
        )
        embed_dim = getattr(self.backbone.encoder, "embed_dim", 256)
        self.head = TerraMindCocoaUPerNetHead(embed_dim=embed_dim, decoder_channels=decoder_channels)

    def set_backbone_freeze(self, freeze: bool) -> None:
        self.backbone.set_freeze(freeze)

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = cocoa_batch_to_terramind_input(batch)
        feat = self.backbone(x)
        return self.head(feat)

    def forward(self, batch: dict[str, Any] | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, dict):
            return self.forward_dict(batch)
        raise TypeError("TerraMindCocoaSegmentation expects a batch dict")

    @torch.no_grad()
    def predict_proba(self, batch: dict[str, Any]) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward_dict(batch))

    @torch.no_grad()
    def predict_proba_numpy(self, batch: dict[str, Any]) -> np.ndarray:
        prob = self.predict_proba(batch)[0, 0].cpu().numpy()
        return prob.astype(np.float32)


class TerraMindTiMCocoaSegmentation(nn.Module):
    """TiM re-encode path + UPerNet cocoa head."""

    def __init__(
        self,
        *,
        tim_modalities: list[str] | None = None,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.tim = TerraMindTiM(pretrained=pretrained_backbone)
        self.tim_modalities = tim_modalities or ["LULC", "NDVI"]
        embed_dim = 256
        self.head = TerraMindCocoaUPerNetHead(embed_dim=embed_dim)

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = cocoa_batch_to_terramind_input(batch)
        feat = self.tim.predict(x, tim_modalities=self.tim_modalities)
        return self.head(feat)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.forward_dict(batch)

    @torch.no_grad()
    def predict_proba_numpy(self, batch: dict[str, Any]) -> np.ndarray:
        self.eval()
        prob = torch.sigmoid(self.forward_dict(batch))[0, 0].cpu().numpy()
        return prob.astype(np.float32)


def load_terramind_seg_checkpoint(
    path: Path | str,
    *,
    device: str = "cpu",
    use_tim: bool = False,
) -> nn.Module:
    """Load fine-tuned TerraMind segmentation weights."""
    path = Path(path)
    cls = TerraMindTiMCocoaSegmentation if use_tim else TerraMindCocoaSegmentation
    model = cls()
    if path.is_file():
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        logger.warning("TerraMind checkpoint missing at %s; using random init", path)
    model.eval()
    return model


__all__ = [
    "DEFAULT_TERRAMIND_CHECKPOINT",
    "DEFAULT_TERRAMIND_TIM_CHECKPOINT",
    "TerraMindCocoaSegmentation",
    "TerraMindTiMCocoaSegmentation",
    "load_terramind_seg_checkpoint",
]
