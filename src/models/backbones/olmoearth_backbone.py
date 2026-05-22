"""
OlmoEarth geospatial encoder (Allen AI, Apache-2.0).

Loads HuggingFace checkpoints when available; falls back to a lightweight
conv stem for CI and local dev without downloaded weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

OlmoEarthSize = Literal["nano", "tiny", "base", "large"]

HF_REPO_BY_SIZE: dict[str, str] = {
    "nano": "allenai/OlmoEarth-Nano",
    "tiny": "allenai/OlmoEarth-Tiny",
    "base": "allenai/OlmoEarth-Base",
    "large": "allenai/OlmoEarth-Large",
}

EMBED_DIM_BY_SIZE: dict[str, int] = {
    "nano": 128,
    "tiny": 256,
    "base": 512,
    "large": 768,
}


class _StubOlmoEncoder(nn.Module):
    """Deterministic conv stem when HF weights are unavailable."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class OlmoEarthBackbone(nn.Module):
    """
    OlmoEarth encoder with Galileo-compatible ``encode`` / ``encode_parcel`` API.

    Parameters
    ----------
    model_size:
        One of ``nano``, ``tiny``, ``base``, ``large``.
    freeze:
        Freeze encoder weights when True.
    use_hf:
        Attempt HuggingFace ``AutoModel`` load; on failure use stub encoder.
    """

    def __init__(
        self,
        model_size: OlmoEarthSize = "base",
        *,
        freeze: bool = True,
        use_hf: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if model_size not in HF_REPO_BY_SIZE:
            raise ValueError(f"model_size must be one of {sorted(HF_REPO_BY_SIZE)}")
        self.model_size = model_size
        self.embed_dim = EMBED_DIM_BY_SIZE[model_size]
        self._hf_model: nn.Module | None = None
        self._stub = _StubOlmoEncoder(self.embed_dim)
        if use_hf:
            self._try_load_hf(cache_dir)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _try_load_hf(self, cache_dir: str | Path | None) -> None:
        try:
            from transformers import AutoModel

            repo = HF_REPO_BY_SIZE[self.model_size]
            self._hf_model = AutoModel.from_pretrained(
                repo,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=True,
            )
        except Exception:
            self._hf_model = None

    def _tensor_from_stacks(
        self,
        s2_stack: torch.Tensor,
        s1_stack: torch.Tensor,
        era5_stack: torch.Tensor,
        dem: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal stacks to ``[B, C, H, W]`` for the stub/HF path."""
        b, t, h, w, _ = s2_stack.shape
        s2_mean = s2_stack.mean(dim=1).permute(0, 3, 1, 2)
        s1_mean = s1_stack.mean(dim=1).permute(0, 3, 1, 2)
        if era5_stack.ndim == 3:
            era = era5_stack.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w)
        else:
            era = era5_stack.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w)
        era = era[:, : min(4, era.shape[1])]
        dem_t = dem.permute(0, 3, 1, 2) if dem.ndim == 4 else dem
        parts = [s2_mean[:, :6], s1_mean, era, dem_t[:, :2]]
        x = torch.cat(parts, dim=1)
        if x.shape[1] < 16:
            pad = torch.zeros(b, 16 - x.shape[1], h, w, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    def encode(
        self,
        s2_stack: torch.Tensor,
        s1_stack: torch.Tensor,
        era5_stack: torch.Tensor,
        dem: torch.Tensor,
    ) -> torch.Tensor:
        """Dense feature map ``[B, D, H', W']``."""
        x = self._tensor_from_stacks(s2_stack, s1_stack, era5_stack, dem)
        if self._hf_model is not None:
            out = self._hf_model(pixel_values=x)
            feats = getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
            if feats.ndim == 3:
                side = int(feats.shape[1] ** 0.5)
                feats = feats.transpose(1, 2).reshape(feats.shape[0], -1, side, side)
            return feats
        return self._stub(x)

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        s2 = batch_dict["s2"]
        s1 = batch_dict["s1"]
        era5 = batch_dict["era5"]
        dem = batch_dict["dem"]
        assert s2 is not None and s1 is not None and era5 is not None and dem is not None
        return self.encode(s2, s1, era5, dem)

    def encode_parcel(self, parcel_inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
        feat = self.forward(parcel_inputs)
        return feat.mean(dim=(-2, -1))


__all__ = ["OlmoEarthBackbone", "HF_REPO_BY_SIZE", "EMBED_DIM_BY_SIZE"]
