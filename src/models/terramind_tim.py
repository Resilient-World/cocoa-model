"""
Thinking-in-Modalities (TiM) inference path for TerraMind 1.0.

Generates intermediate token modalities (LULC, NDVI) from S2L2A, concatenates with
inputs, and re-encodes for downstream segmentation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

from models.terramind_backbone import (
    DEFAULT_MODALITIES,
    DEFAULT_TIM_VARIANT,
    ModalityKey,
    TerraMindBackbone,
    _collapse_tokens_to_map,
    _load_terramind_encoder,
)

logger = logging.getLogger(__name__)

SUPPORTED_TIM_MODALITIES = frozenset({"LULC", "NDVI"})


class TerraMindTiM(nn.Module):
    """
    TiM facade: generate missing modalities then re-encode.

    Uses TerraTorch ``terramind_v1_base_tim`` when available; otherwise applies a
    lightweight proxy that derives pseudo-token maps from S2L2A statistics.
    """

    def __init__(
        self,
        *,
        variant: str = DEFAULT_TIM_VARIANT,
        pretrained: bool = False,
        reencode_variant: str = "terramind_v1_base",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.tim_encoder = _load_terramind_encoder(variant, pretrained=pretrained, tim=True)
        self.reencoder = TerraMindBackbone(variant=reencode_variant, freeze=True, pretrained=pretrained)
        self._use_proxy = isinstance(self.tim_encoder, nn.Module) and not hasattr(
            self.tim_encoder, "generate"
        )

    @staticmethod
    def _validate_modalities(tim_modalities: Sequence[str]) -> list[str]:
        out = [str(m).upper() for m in tim_modalities]
        bad = [m for m in out if m not in SUPPORTED_TIM_MODALITIES]
        if bad:
            raise ValueError(f"tim_modalities must be subset of {sorted(SUPPORTED_TIM_MODALITIES)}; got {bad}")
        return out

    def _generate_modality_proxy(self, s2: torch.Tensor, name: str) -> torch.Tensor:
        """Derive pseudo LULC / NDVI maps from S2 (proxy when TerraTorch TiM unavailable)."""
        if s2.dim() == 5:
            s2 = s2.mean(dim=1)
        nir = s2[:, 7:8] if s2.shape[1] >= 8 else s2[:, -1:]
        red = s2[:, 3:4] if s2.shape[1] >= 4 else s2[:, :1]
        if name == "NDVI":
            ndvi = (nir - red) / (nir + red + 1e-6)
            return ndvi.clamp(-1, 1)
        # LULC proxy: edge-heavy classes from band std
        std = s2.std(dim=1, keepdim=True)
        return (std / (std.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values + 1e-6)).clamp(
            0, 1
        )

    def _generate_modality(self, x: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if hasattr(self.tim_encoder, "generate"):
            try:
                gen = self.tim_encoder.generate({k: x.get(k) for k in x if k in DEFAULT_MODALITIES}, [name])
                if isinstance(gen, dict) and name in gen:
                    return _collapse_tokens_to_map(gen[name])
            except Exception as exc:
                logger.debug("TerraMind TiM generate failed for %s: %s", name, exc)
        if "S2L2A" not in x:
            raise KeyError("TiM generation requires S2L2A in input dict")
        return self._generate_modality_proxy(x["S2L2A"], name)

    def predict(
        self,
        x: dict[str, torch.Tensor],
        tim_modalities: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Run TiM then re-encode to a feature map for the cocoa head.

        Parameters
        ----------
        x:
            Input modalities (at least ``S2L2A``; ``S1GRD`` / ``DEM`` optional).
        tim_modalities:
            Token modalities to synthesize before re-encoding (default LULC + NDVI).

        Returns
        -------
        torch.Tensor
            Feature map ``[B, D, H', W']``.
        """
        mods = self._validate_modalities(tim_modalities or ["LULC", "NDVI"])
        enriched = dict(x)
        ref = enriched.get("S2L2A")
        if ref is None:
            raise KeyError("TiM predict requires S2L2A")
        s2_parts = [ref]
        for name in mods:
            generated = self._generate_modality(enriched, name)
            if generated.shape[-2:] != ref.shape[-2:]:
                generated = torch.nn.functional.interpolate(
                    generated, size=ref.shape[-2:], mode="bilinear", align_corners=False
                )
            s2_parts.append(generated)
        enriched["S2L2A"] = torch.cat(s2_parts, dim=1)
        return self.reencoder(enriched)


__all__ = ["SUPPORTED_TIM_MODALITIES", "TerraMindTiM"]
