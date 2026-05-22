"""
Phenology-Aware Positional Encoding (PAPE) for climate sequences.

Adds a learnable residual delta to ``[B, T, F]`` climate tensors conditioned on
day-of-year, regional crop stage, and region embedding. Final linear layer is
zero-initialized so v1 checkpoints behave identically at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from data.cocoa_exposure import REGIONS, normalize_region_key

STAGE_NAMES: tuple[str, ...] = (
    "dry_season",
    "main_crop_setting",
    "main_crop_pod_dev",
    "mid_crop",
)

REGION_KEYS: tuple[str, ...] = tuple(REGIONS.keys())
REGION_TO_ID: dict[str, int] = {k: i for i, k in enumerate(REGION_KEYS)}

_DEFAULT_PHENOLOGY_PATH = Path(__file__).resolve().parents[3] / "config" / "phenology.yaml"


@dataclass(frozen=True)
class PhenologyConfig:
    """Per-region DOY stage intervals from ``config/phenology.yaml``."""

    regions: dict[str, dict[str, list[list[int]]]]


def load_phenology_config(path: str | Path | None = None) -> PhenologyConfig:
    """Load regional phenology calendars from YAML."""
    cfg_path = Path(path) if path is not None else _DEFAULT_PHENOLOGY_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    regions = raw.get("regions", raw)
    return PhenologyConfig(regions=regions)


def region_to_id(region_key: str) -> int:
    """Map a :data:`~data.cocoa_exposure.REGIONS` key to ``{0..7}``."""
    return REGION_TO_ID[normalize_region_key(region_key)]


def _doy_in_interval(doy: int, start: int, end: int) -> bool:
    """Inclusive start, exclusive end; supports wrap (start > end)."""
    d = int(doy)
    if start < end:
        return start <= d < end
    return d >= start or d < end


def _stage_index_for_doy(
    intervals: dict[str, list[list[int]]],
    doy: int,
) -> int:
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        for start, end in intervals.get(stage_name, []):
            if _doy_in_interval(doy, int(start), int(end)):
                return stage_idx
    return 0


def crop_stage_one_hot(
    region_key: str,
    doy: int | np.ndarray,
    *,
    config: PhenologyConfig | None = None,
) -> np.ndarray:
    """
    One-hot crop stage for a region and DOY.

    Returns shape ``[4]`` for scalar ``doy`` or ``[T, 4]`` for a DOY vector.
    """
    cfg = config or load_phenology_config()
    key = normalize_region_key(region_key)
    intervals = cfg.regions[key]

    if np.ndim(doy) == 0:
        idx = _stage_index_for_doy(intervals, int(doy))
        out = np.zeros(len(STAGE_NAMES), dtype=np.float32)
        out[idx] = 1.0
        return out

    doys = np.asarray(doy, dtype=np.int32).reshape(-1)
    out = np.zeros((doys.size, len(STAGE_NAMES)), dtype=np.float32)
    for t, d in enumerate(doys):
        out[t, _stage_index_for_doy(intervals, int(d))] = 1.0
    return out


class PhenologyAwarePositionalEncoding(nn.Module):
    """
    Residual phenology encoding on climate tensors.

    ``climate_out = climate + MLP(cond)`` with ``cond`` = DOY Fourier (2) +
    crop stage one-hot (4) + region embedding (8).
    """

    def __init__(
        self,
        n_features: int,
        *,
        n_regions: int = len(REGION_KEYS),
        hidden: int = 48,
        phenology_config: PhenologyConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_regions = n_regions
        self.cond_dim = 2 + len(STAGE_NAMES) + 8
        self._phenology = phenology_config or load_phenology_config()

        self.region_embed = nn.Embedding(n_regions, 8)
        self.mlp = nn.Sequential(
            nn.Linear(self.cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _stage_tensor(
        self,
        region_id: Tensor,
        doy: Tensor,
    ) -> Tensor:
        """Build ``[B, T, 4]`` stage one-hot from region ids and DOY."""
        b, t = doy.shape
        stages = torch.zeros(b, t, len(STAGE_NAMES), device=doy.device, dtype=doy.dtype)
        rid = region_id.detach().cpu().numpy()
        doy_np = doy.detach().cpu().numpy()
        for bi in range(b):
            region_key = REGION_KEYS[int(rid[bi])]
            one_hot = crop_stage_one_hot(region_key, doy_np[bi], config=self._phenology)
            stages[bi] = torch.from_numpy(one_hot).to(device=doy.device, dtype=doy.dtype)
        return stages

    def forward(
        self,
        climate: Tensor,
        region_id: Tensor,
        *,
        doy: Tensor | None = None,
    ) -> Tensor:
        """
        Apply PAPE residual to ``climate`` ``[B, T, F]``.

        Parameters
        ----------
        climate:
            Daily climate features.
        region_id:
            ``[B]`` long tensor in ``{0..n_regions-1}``.
        doy:
            Optional ``[B, T]`` day-of-year; defaults to ``1..T`` per batch row.
        """
        if climate.ndim != 3:
            raise ValueError(f"climate must be [B, T, F], got {tuple(climate.shape)}")
        b, t, f = climate.shape
        if f != self.n_features:
            raise ValueError(f"expected F={self.n_features}, got {f}")
        if region_id.shape != (b,):
            raise ValueError(f"region_id must be [B], got {tuple(region_id.shape)}")

        if doy is None:
            doy = (
                torch.arange(1, t + 1, device=climate.device, dtype=climate.dtype)
                .unsqueeze(0)
                .expand(b, -1)
            )
        elif doy.shape != (b, t):
            raise ValueError(f"doy must be [B, T], got {tuple(doy.shape)}")

        phase = 2.0 * np.pi * doy.float() / 365.0
        doy_sin = torch.sin(phase)
        doy_cos = torch.cos(phase)
        stage_oh = self._stage_tensor(region_id, doy)
        region_emb = self.region_embed(region_id.long()).unsqueeze(1).expand(-1, t, -1)

        cond = torch.cat(
            [
                doy_sin.unsqueeze(-1),
                doy_cos.unsqueeze(-1),
                stage_oh,
                region_emb,
            ],
            dim=-1,
        )
        delta = self.mlp(cond)
        return climate + delta

    def delta(
        self,
        climate: Tensor,
        region_id: Tensor,
        *,
        doy: Tensor | None = None,
    ) -> Tensor:
        """Return only the PAPE residual (for identity checks)."""
        encoded = self.forward(climate, region_id, doy=doy)
        return encoded - climate


__all__ = [
    "REGION_KEYS",
    "REGION_TO_ID",
    "STAGE_NAMES",
    "PhenologyAwarePositionalEncoding",
    "PhenologyConfig",
    "crop_stage_one_hot",
    "load_phenology_config",
    "region_to_id",
]
