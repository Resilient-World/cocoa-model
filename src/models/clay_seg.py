"""Clay v1.5 cocoa segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.backbones.clay_backbone import ClayBackbone, DEFAULT_EMBED_DIM
from models.backbones.clay_cocoa_head import ClayCocoaSegHead


class ClayCocoaSegmentation(nn.Module):
    def __init__(self, *, freeze_backbone: bool = True, out_size: tuple[int, int] = (64, 64), use_hf: bool = True) -> None:
        super().__init__()
        self.backbone = ClayBackbone(freeze=freeze_backbone, use_hf=use_hf)
        self.head = ClayCocoaSegHead(embed_dim=DEFAULT_EMBED_DIM, out_size=out_size)

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        return self.head(self.backbone(batch_dict))

    @torch.no_grad()
    def predict_proba_numpy(self, batch_dict: dict[str, torch.Tensor | None]):
        import numpy as np

        self.eval()
        prob = torch.sigmoid(self.forward(batch_dict))[0, 0].cpu().numpy()
        return prob.astype(np.float32)


def load_clay_seg_checkpoint(path: Path | str, device: str = "cpu") -> ClayCocoaSegmentation:
    model = ClayCocoaSegmentation(use_hf=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
