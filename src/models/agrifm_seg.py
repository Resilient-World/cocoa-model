"""
AgriFM Video Swin backbone + versatile decoder for binary cocoa maps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.agrifm_backbone import AgriFMBackbone, Modality
from models.agrifm_cocoa_head import AgriFMCocoaSegHead

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGRIFM_PRETRAINED = _REPO_ROOT / "models" / "agrifm" / "agrifm_s2_pretrained.pt"
DEFAULT_AGRIFM_FINETUNED = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"
DEFAULT_AGRIFM_CHECKPOINT = DEFAULT_AGRIFM_FINETUNED
S2_BAND_COUNT = 10


class AgriFMCocoaSegmentation(nn.Module):
    """
    AgriFM encoder + :class:`AgriFMCocoaSegHead` for per-pixel P(cocoa).

    Parameters
    ----------
    checkpoint_path:
        AgriFM S2 pretrained weights (backbone only unless full seg checkpoint).
    modality:
        Sensor preset passed to the backbone.
    freeze_backbone:
        Freeze encoder weights during fine-tuning.
    out_size:
        Output map resolution.
    num_frames:
        Fixed temporal length; ``None`` infers from input.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_AGRIFM_CHECKPOINT,
        modality: Modality = "S2",
        freeze_backbone: bool = True,
        out_size: tuple[int, int] = (256, 256),
        num_frames: int | None = None,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = AgriFMBackbone(
            checkpoint_path=checkpoint_path,
            modality=modality,
            freeze=freeze_backbone,
            num_frames=num_frames,
        )
        self.head = AgriFMCocoaSegHead(out_size=out_size, dropout_p=head_dropout)
        self.out_size = out_size

    def set_backbone_freeze(self, freeze: bool) -> None:
        """Toggle gradient flow through the Video Swin encoder."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = not freeze

    @staticmethod
    def s2_batch_to_tensor(s2: torch.Tensor) -> torch.Tensor:
        """
        Convert benchmark tile ``[B,T,H,W,C]`` or ``[B,C,T,H,W]`` to ``[B,C,T,H,W]``.
        """
        if s2.dim() != 5:
            raise ValueError(f"Expected 5D S2 tensor, got {tuple(s2.shape)}")
        if s2.shape[-1] == S2_BAND_COUNT:
            return s2.permute(0, 4, 1, 2, 3).contiguous()
        if s2.shape[1] == S2_BAND_COUNT:
            return s2
        raise ValueError(f"Cannot infer channel dim for S2 shape {tuple(s2.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stages = self.backbone(x)
        return self.head(stages)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(cocoa) ``[B,1,H,W]``."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_proba_numpy(self, s2: torch.Tensor) -> np.ndarray:
        """Return ``[H,W]`` probabilities for batch size 1."""
        x = self.s2_batch_to_tensor(s2)
        prob = self.predict_proba(x)[0, 0].cpu().numpy()
        return prob.astype(np.float32)


def load_agrifm_seg_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
    modality: Modality = "S2",
    out_size: tuple[int, int] = (256, 256),
    pretrained_path: Path | None = None,
) -> AgriFMCocoaSegmentation:
    """Load fine-tuned segmentation; falls back to backbone-only pretrained weights."""
    backbone_ckpt = pretrained_path or DEFAULT_AGRIFM_PRETRAINED
    if checkpoint_path.is_file():
        backbone_ckpt = checkpoint_path
    model = AgriFMCocoaSegmentation(
        checkpoint_path=backbone_ckpt,
        modality=modality,
        out_size=out_size,
    )
    if checkpoint_path.is_file():
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        if isinstance(state, dict) and any(
            k.startswith("head.") or k.startswith("backbone.") for k in state
        ):
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
