"""Tests for :mod:`analysis.climate_attribution`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch import Tensor, nn

from analysis.climate_attribution import (
    climate_attributable_loss,
    decompose_avoided_loss,
)

EXPECTED_COLUMNS = {
    "farm_id",
    "year",
    "y_factual_mean",
    "y_factual_std",
    "y_cf_mean",
    "y_cf_std",
    "climate_loss_tpha",
    "climate_loss_se",
}


def _synthetic_climate_grids(
    *,
    year: int = 2020,
    tmax_base: float = 28.0,
    tmax_delta: float = 0.0,
    precip_scale: float = 1.0,
) -> xr.Dataset:
    time = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    lat = np.array([6.0, 7.0])
    lon = np.array([-5.0, -4.0])
    rng = np.random.default_rng(0)
    seasonal = np.sin(2 * np.pi * np.arange(365) / 365.0)

    def expand_daily(series: np.ndarray) -> np.ndarray:
        return series[:, np.newaxis, np.newaxis] * np.ones((365, 2, 2), dtype=np.float32)

    tmax = tmax_base + 2.0 * seasonal + rng.normal(0, 0.3, 365) + tmax_delta
    tmin = tmax - 6.0
    precip = np.clip(rng.gamma(2.0, 2.0, 365) * precip_scale, 0.0, 40.0)
    srad = 15.0 + 3.0 * seasonal + rng.normal(0, 0.2, 365)

    return xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), expand_daily(tmax.astype(np.float32))),
            "tmin": (("time", "lat", "lon"), expand_daily(tmin.astype(np.float32))),
            "precip": (("time", "lat", "lon"), expand_daily(precip.astype(np.float32))),
            "srad": (("time", "lat", "lon"), expand_daily(srad.astype(np.float32))),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


class StubYieldModel(nn.Module):
    """
    Deterministic yield: hotter / wetter sites score higher.

    With factual +1.5 °C vs counterfactual, ``y_factual > y_cf`` so
    ``climate_loss_tpha = y_factual - y_cf`` is positive (plumbing check).
    """

    def __init__(self, a_tmax: float = 0.15, b_precip: float = 0.002) -> None:
        super().__init__()
        self.a_tmax = a_tmax
        self.b_precip = b_precip

    def eval(self) -> StubYieldModel:
        return self

    def to(self, device: str | torch.device) -> StubYieldModel:
        return self

    def forward(self, climate: Tensor, static: Tensor) -> Tensor:
        del static
        tmax_mean = climate[..., 0].mean(dim=1)
        precip_sum = climate[..., 2].sum(dim=1)
        return self.a_tmax * tmax_mean + self.b_precip * precip_sum


def test_climate_attributable_loss_positive_when_factual_warmer() -> None:
    # Factual +1.5 °C; counterfactual +10% precip (stub uses tmax only so sign is clear)
    factual = _synthetic_climate_grids(tmax_delta=1.5, precip_scale=1.0)
    counterfactual = _synthetic_climate_grids(tmax_delta=0.0, precip_scale=1.1)

    farm_coords = pd.DataFrame(
        {
            "farm_id": ["F1", "F2"],
            "lat": [6.0, 7.0],
            "lon": [-5.0, -4.0],
            "year": [2020, 2020],
        }
    )
    static_features = np.zeros((2, 10), dtype=np.float32)
    static_features[:, 0] = 150.0

    model = StubYieldModel(a_tmax=0.2, b_precip=0.0)
    df = climate_attributable_loss(
        factual,
        counterfactual,
        model,
        static_features,
        farm_coords,
        n_mc_samples=5,
        device="cpu",
    )

    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2
    assert float(df["climate_loss_tpha"].mean()) > 0.0
    assert (df["climate_loss_se"] >= 0.0).all()


def test_decompose_avoided_loss_sensible_bounds() -> None:
    climate_df = pd.DataFrame(
        {
            "farm_id": ["F1", "F2", "F3"],
            "climate_loss_tpha": [0.2, 0.3, 0.25],
        }
    )
    did_att = 0.35
    did_ci = (0.10, 0.60)

    out = decompose_avoided_loss(did_att, did_ci, climate_df, n_bootstrap=500, random_state=0)

    assert out["intervention_att"] == pytest.approx(did_att)
    assert out["intervention_att_ci"] == did_ci
    assert out["climate_attributable_mean"] == pytest.approx(0.25, rel=1e-5)
    lo, hi = out["climate_attributable_ci"]
    assert lo <= out["climate_attributable_mean"] <= hi
    assert out["total_avoided_loss"] == pytest.approx(did_att + 0.25, rel=1e-5)
