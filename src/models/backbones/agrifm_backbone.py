"""
AgriFM Video Swin backbone for multi-scale spatiotemporal features.

Provenance: Li et al. (RSE 2026; arXiv:2505.21357). Pretrained weights from the
AgriFM project (Apache-2.0) are loaded into this MIT reimplementation — the upstream
Python package is not imported at runtime.
"""

from __future__ import annotations

import structlog

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from models.backbones.agrifm_video_swin import PretrainingSwinTransformer3DEncoder, build_agrifm_encoder

log = structlog.get_logger(__name__)

Modality = Literal["S2", "Landsat", "MODIS"]
MIN_FRAMES = 3
MAX_FRAMES = 32
PATCH_TEMPORAL = 4


@dataclass(frozen=True)
class ModalityConfig:
    in_chans: int
    checkpoint_prefix: str


MODALITY_CONFIG: dict[Modality, ModalityConfig] = {
    "S2": ModalityConfig(in_chans=10, checkpoint_prefix="S2_patch_emd"),
    "Landsat": ModalityConfig(in_chans=6, checkpoint_prefix="HLSL30_patch_emd"),
    "MODIS": ModalityConfig(in_chans=7, checkpoint_prefix="Modis_patch_emd"),
}


def _pad_temporal(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad ``[B,C,T,H,W]`` along time to ``target_t`` (replicate last frame)."""
    b, c, t, h, w = x.shape
    if t >= target_t:
        return x[:, :, :target_t]
    pad = target_t - t
    last = x[:, :, -1:].expand(b, c, pad, h, w)
    return torch.cat([x, last], dim=2)


def _features_list_to_stages(
    features_list: list[torch.Tensor],
    base_embed: int,
) -> dict[str, torch.Tensor]:
    """Convert AgriFM ``features_list`` entries to ``[B,T,H,W,C]`` stage tensors."""
    stages: dict[str, torch.Tensor] = {}
    for i, feat in enumerate(features_list):
        b, cd, h, w = feat.shape
        channels = base_embed * (2**i)
        if cd % channels != 0:
            raise ValueError(f"Stage {i}: cannot split {cd} channels into T×{channels}")
        t = cd // channels
        tensor = feat.view(b, channels, t, h, w)
        stages[f"stage_{i + 1}"] = rearrange(tensor, "b c t h w -> b t h w c")
    return stages


def _remap_agrifm_state_dict(
    state_dict: dict[str, torch.Tensor],
    modality: Modality,
) -> dict[str, torch.Tensor]:
    """Map AgriFM checkpoint keys onto :class:`PretrainingSwinTransformer3DEncoder` names."""
    cfg = MODALITY_CONFIG[modality]
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith(cfg.checkpoint_prefix):
            k = k.replace(cfg.checkpoint_prefix, "patch_embed", 1)
        if k.startswith("encoder."):
            k = k.replace("encoder.", "", 1)
        remapped[k] = value
    return remapped


def _log_checkpoint_skips(
    missing: list[str],
    unexpected: list[str],
) -> None:
    payload = {"missing_keys": missing, "unexpected_keys": unexpected}
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_text(json.dumps(payload, indent=2), "agrifm_checkpoint_skipped_keys.json")
            return
    except ImportError:
        pass
    if missing or unexpected:
        log.warning(
            "AgriFM checkpoint partial load: %d missing, %d unexpected keys",
            len(missing),
            len(unexpected),
        )


class AgriFMBackbone(nn.Module):
    """
    Frozen-by-default AgriFM Video Swin encoder returning four pyramid stages.

    Parameters
    ----------
    checkpoint_path:
        Path to AgriFM ``.pt`` / ``.pth`` weights (Apache-2.0).
    modality:
        Sensor preset controlling input channel count and checkpoint key prefix.
    freeze:
        If True, disable gradients on encoder parameters.
    num_frames:
        Expected temporal length. When ``None``, inferred from ``forward`` input
        (must be in ``[3, 32]``). Temporal length is padded to a multiple of 4
        for patch embedding.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        modality: Modality = "S2",
        freeze: bool = True,
        num_frames: int | None = None,
    ) -> None:
        super().__init__()
        if modality not in MODALITY_CONFIG:
            raise ValueError(f"modality must be one of {sorted(MODALITY_CONFIG)}")
        cfg = MODALITY_CONFIG[modality]
        self.modality = modality
        self.in_chans = cfg.in_chans
        self.embed_dim = 128
        self.freeze = freeze
        self._num_frames = num_frames
        self.encoder: PretrainingSwinTransformer3DEncoder = build_agrifm_encoder(
            in_chans=cfg.in_chans,
            embed_dim=self.embed_dim,
        )
        if checkpoint_path.is_file():
            self._load_checkpoint(checkpoint_path)
        else:
            log.warning("AgriFM checkpoint not found at %s; using random init", checkpoint_path)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def num_frames(self) -> int | None:
        return self._num_frames

    def resolve_num_frames(self, temporal: int) -> int:
        """Validate and optionally pad temporal length to a multiple of ``PATCH_TEMPORAL``."""
        if temporal < MIN_FRAMES or temporal > MAX_FRAMES:
            raise ValueError(f"Temporal length must be in [{MIN_FRAMES}, {MAX_FRAMES}], got {temporal}")
        target = self._num_frames if self._num_frames is not None else temporal
        if target < MIN_FRAMES or target > MAX_FRAMES:
            raise ValueError(f"num_frames must be in [{MIN_FRAMES}, {MAX_FRAMES}], got {target}")
        pad_t = ((target + PATCH_TEMPORAL - 1) // PATCH_TEMPORAL) * PATCH_TEMPORAL
        return max(pad_t, temporal if self._num_frames is None else pad_t)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
        else:
            state_dict = raw if isinstance(raw, dict) else {}
        state_dict = _remap_agrifm_state_dict(state_dict, self.modality)
        result = self.encoder.load_state_dict(state_dict, strict=False)
        _log_checkpoint_skips(list(result.missing_keys), list(result.unexpected_keys))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Sentinel (or other) stack ``[B, C, T, H, W]``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys ``stage_1`` … ``stage_4`` with tensors ``[B, T_i, H_i, W_i, C_i]``.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B,C,T,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels for {self.modality}, got {x.shape[1]}")
        t_in = x.shape[2]
        target_t = self.resolve_num_frames(t_in)
        if x.shape[2] != target_t:
            x = _pad_temporal(x, target_t)
        feats = self.encoder(x)
        features_list = feats["features_list"]
        if len(features_list) < 4:
            raise RuntimeError(f"Expected 4 feature stages, got {len(features_list)}")
        return _features_list_to_stages(features_list[:4], self.embed_dim)
