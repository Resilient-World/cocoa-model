"""OlmoEarth cocoa segmentation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.backbones.olmoearth_backbone import EMBED_DIM_BY_SIZE, OlmoEarthBackbone
from models.backbones.olmoearth_cocoa_head import OlmoEarthCocoaSegHead


class OlmoEarthCocoaSegmentation(nn.Module):
    def __init__(
        self,
        model_size: str = "base",
        *,
        freeze_backbone: bool = True,
        out_size: tuple[int, int] = (64, 64),
        use_hf: bool = True,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.backbone = OlmoEarthBackbone(model_size=model_size, freeze=freeze_backbone, use_hf=use_hf)
        dim = EMBED_DIM_BY_SIZE[model_size]
        self.head = OlmoEarthCocoaSegHead(embed_dim=dim, out_size=out_size)

    @staticmethod
    def build_batch_dict(
        *,
        s2: torch.Tensor,
        s1: torch.Tensor,
        era5: torch.Tensor,
        dem: torch.Tensor,
        location: torch.Tensor,
        months: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"s2": s2, "s1": s1, "era5": era5, "dem": dem, "location": location, "months": months}

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        feats = self.backbone(batch_dict)
        return self.head(feats)

    @torch.no_grad()
    def predict_proba(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(batch_dict))

    @torch.no_grad()
    def predict_proba_numpy(self, batch_dict: dict[str, torch.Tensor | None]) -> Any:
        import numpy as np

        prob = self.predict_proba(batch_dict)[0, 0].cpu().numpy()
        return prob.astype(np.float32)


def load_olmoearth_seg_checkpoint(
    path: Path | str,
    *,
    model_size: str = "base",
    device: str = "cpu",
) -> OlmoEarthCocoaSegmentation:
    model = OlmoEarthCocoaSegmentation(model_size=model_size, use_hf=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
