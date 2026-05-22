"""
Galileo-Base binary cocoa segmentation for 10 m P(cocoa) maps.

Multimodal inputs (Sentinel-2 10 bands, Sentinel-1 VV/VH, ERA5 monthly stack,
DEM) are encoded via :class:`~models.backbones.galileo_backbone.GalileoCocoaBackbone` and
decoded to per-pixel logits. Intended supervision mix:

- **Weak:** FDP 2025a probability rasters (Forest Data Partnership)
- **Strong:** Kalischek et al. (2023) in-situ validation tiles over CIV/Ghana

Fine-tuning is driven by :mod:`training.train_galileo_cocoa` or a dedicated
trainer that supplies ``batch_dict`` keys documented in
:func:`data.utils.cocoa_batch_to_galileo_input`.
"""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.galileo_backbone import GalileoCocoaBackbone

log = structlog.get_logger(__name__)

# Input channel contract (10 m native stack)
S2_BAND_COUNT = 10  # Sentinel-2 optical stack (B2–B12 subset used in cocoa pipeline)
S1_BAND_COUNT = 2  # VV, VH
# Full monthly stack used in ingest / training rasters (5 channels)
ERA5_MONTHLY_VARS = ("precip", "temperature_2m", "rh", "vpd", "et0")
ERA5_MONTHLY_COUNT = len(ERA5_MONTHLY_VARS)  # 5
# Galileo encoder time bands (see :mod:`models.backbones.vendor.galileo_data_utils`)
ERA5_GALILEO_VARS = ("precip", "temperature_2m")
ERA5_GALILEO_COUNT = len(ERA5_GALILEO_VARS)  # 2
DEM_BAND_COUNT = 2  # elevation, slope


class GalileoCocoaSegmentation(nn.Module):
    """
    Galileo-Base ViT backbone + FPN head → single-channel P(cocoa).

    Parameters
    ----------
    model_size:
        Galileo HF size folder (``base`` recommended).
    patch_size:
        Flexi patch size passed to the encoder (4 for 10 m tiles).
    freeze_backbone:
        Start with frozen encoder weights (unfreeze for stage-2 fine-tune).
    """

    def __init__(
        self,
        model_size: str = "base",
        patch_size: int = 4,
        freeze_backbone: bool = True,
        *,
        decoder_channels: int = 128,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.patch_size = patch_size
        self.backbone = GalileoCocoaBackbone(
            model_size=model_size,
            freeze=freeze_backbone,
            patch_size=patch_size,
            normalize=normalize,
            cache_dir=cache_dir,
        )
        self._head: nn.Module | None = None
        self._decoder_channels = decoder_channels

    def set_backbone_freeze(self, freeze: bool) -> None:
        self.backbone.set_freeze(freeze)

    def _ensure_head(self, embed_dim: int, device: torch.device) -> nn.Module:
        if self._head is not None:
            return self._head
        mid = max(embed_dim // 2, 32)
        self._head = nn.Sequential(
            nn.Conv2d(embed_dim, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
        ).to(device)
        return self._head

    @staticmethod
    def _squeeze_leading_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 1 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    @staticmethod
    def _era5_for_galileo(era5: torch.Tensor) -> torch.Tensor:
        """Map monthly stack to Galileo ``[T, 2]`` (precip + temperature_2m)."""
        if era5.dim() == 5:
            era5 = era5[0, 0]  # [T, C] from [B, T, H, W, C]
        elif era5.dim() == 4:
            era5 = era5[0] if era5.shape[0] == 1 else era5
        if era5.dim() == 3 and era5.shape[0] == 1:
            era5 = era5.squeeze(0)
        if era5.shape[-1] > ERA5_GALILEO_COUNT:
            era5 = era5[..., : ERA5_GALILEO_COUNT]
        if era5.dim() != 2:
            raise ValueError(f"era5 must reduce to [T, {ERA5_GALILEO_COUNT}]; got {era5.shape}")
        return era5

    @staticmethod
    def build_batch_dict(
        *,
        s2: torch.Tensor,
        s1: torch.Tensor,
        era5: torch.Tensor,
        dem: torch.Tensor,
        location: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Pack modality tensors into a Galileo ``batch_dict``.

        Accepts optional leading batch dim ``B=1``. Space-time tensors use
        ``[T, H, W, C]`` internally (Galileo ``[H, W, T, C]`` after conversion).
        """
        s2_out = GalileoCocoaSegmentation._squeeze_leading_batch(s2)
        s1_out = GalileoCocoaSegmentation._squeeze_leading_batch(s1)
        dem_out = GalileoCocoaSegmentation._squeeze_leading_batch(dem)
        era5_out = GalileoCocoaSegmentation._era5_for_galileo(era5)
        loc_out = (
            GalileoCocoaSegmentation._squeeze_leading_batch(location)
            if location is not None
            else None
        )
        months_out = (
            GalileoCocoaSegmentation._squeeze_leading_batch(months)
            if months is not None
            else None
        )
        return {
            "s2": s2_out,
            "s1": s1_out,
            "era5": era5_out,
            "srtm": dem_out,
            "location": loc_out,
            "months": months_out,
        }

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Logits ``[B, 1, H, W]`` (upsampled to input resolution).
        """
        features = self.backbone(batch_dict)
        head = self._ensure_head(self.backbone.embedding_size, features.device)
        logits = head(features)
        if self.patch_size > 1:
            logits = F.interpolate(
                logits,
                scale_factor=float(self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def predict_proba(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """Sigmoid probability map ``[B, 1, H, W]`` in [0, 1]."""
        return torch.sigmoid(self.forward(batch_dict))

    @torch.no_grad()
    def predict_proba_numpy(
        self,
        batch_dict: dict[str, torch.Tensor | None],
    ) -> Any:
        """CPU numpy probability map ``[H, W]`` for batch size 1."""
        import numpy as np

        self.eval()
        prob = self.predict_proba(batch_dict).squeeze(0).squeeze(0)
        return prob.detach().cpu().numpy().astype(np.float32)

    def weak_strong_bce_loss(
        self,
        logits: torch.Tensor,
        *,
        fdp_target: torch.Tensor | None = None,
        kalischek_target: torch.Tensor | None = None,
        fdp_weight: float = 0.5,
        kalischek_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Combined BCE for weak (FDP soft) and strong (Kalischek binary) supervision.

        Targets are ``[B, 1, H, W]`` in [0, 1].
        """
        loss = logits.new_tensor(0.0)
        n_terms = 0
        if fdp_target is not None:
            loss = loss + fdp_weight * F.binary_cross_entropy_with_logits(
                logits, fdp_target.clamp(0.0, 1.0)
            )
            n_terms += 1
        if kalischek_target is not None:
            loss = loss + kalischek_weight * F.binary_cross_entropy_with_logits(
                logits, kalischek_target.clamp(0.0, 1.0)
            )
            n_terms += 1
        if n_terms == 0:
            raise ValueError("At least one of fdp_target or kalischek_target is required")
        return loss / max(n_terms, 1)


def load_galileo_seg_checkpoint(
    path: str | Path,
    *,
    model_size: str = "base",
    device: str | torch.device = "cpu",
) -> GalileoCocoaSegmentation:
    """Load fine-tuned binary head + backbone weights."""
    model = GalileoCocoaSegmentation(model_size=model_size, freeze_backbone=True)
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    log.info("Loaded GalileoCocoaSegmentation from %s", path)
    return model


__all__ = [
    "DEM_BAND_COUNT",
    "ERA5_GALILEO_COUNT",
    "ERA5_GALILEO_VARS",
    "ERA5_MONTHLY_COUNT",
    "ERA5_MONTHLY_VARS",
    "GalileoCocoaSegmentation",
    "S1_BAND_COUNT",
    "S2_BAND_COUNT",
    "load_galileo_seg_checkpoint",
]
