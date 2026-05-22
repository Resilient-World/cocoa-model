"""
Galileo feature extractor for cocoa exposure mapping.

Wraps ``construct_galileo_input()`` to build :class:`~models.backbones.vendor.galileo_data_utils.MaskedOutput`
from ingest modules (Sentinel-1, Sentinel-2, ERA5, TerraClimate, SRTM, Dynamic World) and
returns frozen embeddings ready for a linear probe or LoRA head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .galileo_loader import load_galileo
from .vendor.galileo_data_utils import MaskedOutput, construct_galileo_input


def _thw_to_hwt(tensor: torch.Tensor) -> torch.Tensor:
    """``[T, H, W, C]`` → ``[H, W, T, C]`` (Galileo layout)."""
    return tensor.permute(1, 2, 0, 3)


def _batch_masked(masked: MaskedOutput, device: str) -> MaskedOutput:
    return MaskedOutput(
        **{
            k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
            for k, v in masked._asdict().items()
        }
    )


@dataclass
class GalileoFeatureConfig:
    size: str = "base"
    patch_size: int = 8
    device: str = "cuda"
    normalize: bool = True
    batch_size: int = 16


class GalileoFeatureExtractor:
    def __init__(self, cfg: GalileoFeatureConfig | None = None) -> None:
        self.cfg = cfg or GalileoFeatureConfig()
        self.model = load_galileo(self.cfg.size, device=self.cfg.device)

    @torch.no_grad()
    def embed(
        self,
        *,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        ndvi: torch.Tensor | None = None,
        srtm: torch.Tensor | None = None,
        dw: torch.Tensor | None = None,
        era5: torch.Tensor | None = None,
        terraclim: torch.Tensor | None = None,
        viirs: torch.Tensor | None = None,
        location: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-patch embeddings: ``[N_tokens, D]``."""
        if s2 is not None and s2.ndim == 4:
            s2 = _thw_to_hwt(s2)
        if s1 is not None and s1.ndim == 4:
            s1 = _thw_to_hwt(s1)
        if ndvi is not None and ndvi.ndim == 3:
            ndvi = _thw_to_hwt(ndvi.unsqueeze(-1)).squeeze(-1)

        masked = construct_galileo_input(
            s2=s2,
            s1=s1,
            ndvi=ndvi,
            srtm=srtm,
            dw=dw,
            era5=era5,
            tc=terraclim,
            viirs=viirs,
            latlon=location,
            months=months,
            normalize=self.cfg.normalize,
        )
        batched = _batch_masked(masked, self.cfg.device)

        encoded = self.model(
            batched.space_time_x,
            batched.space_x,
            batched.time_x,
            batched.static_x,
            batched.space_time_mask,
            batched.space_mask,
            batched.time_mask,
            batched.static_mask,
            batched.months,
            self.cfg.patch_size,
        )
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = encoded
        tokens = self.model.apply_mask_and_average_tokens_per_patch(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )
        return tokens.squeeze(0).cpu()
