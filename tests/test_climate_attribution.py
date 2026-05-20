"""Tests for :mod:`analysis.climate_attribution`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch import Tensor, nn

from analysis.climate_attribution import (
    _fao_et0_numpy,
    climate_attributable_loss,
    decompose_avoided_loss,
    extract_daily_climate_11ch,
    extract_daily_climate_4ch,
)
from data.attrici_counterfactual import _fao_et0_array
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES, CLIMATE_IDX, N_CLIMATE_CHANNELS

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
    """ERA5-like daily grid with all variables needed for 11-channel extraction."""
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
    srad = np.maximum(15.0 + 3.0 * seasonal + rng.normal(0, 0.2, 365), 0.1)
    rh = np.clip(70.0 + 10.0 * seasonal + rng.normal(0, 2.0, 365), 5.0, 100.0)
    wind = np.maximum(2.0 + 0.5 * seasonal + rng.normal(0, 0.1, 365), 0.1)
    sm_root = np.clip(0.25 + 0.05 * seasonal, 0.05, 0.45)
    co2 = np.full(365, 415.0, dtype=np.float32)

    return xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), expand_daily(tmax.astype(np.float32))),
            "tmin": (("time", "lat", "lon"), expand_daily(tmin.astype(np.float32))),
            "precip": (("time", "lat", "lon"), expand_daily(precip.astype(np.float32))),
            "srad": (("time", "lat", "lon"), expand_daily(srad.astype(np.float32))),
            "rh_mean": (("time", "lat", "lon"), expand_daily(rh.astype(np.float32))),
            "wind10m": (("time", "lat", "lon"), expand_daily(wind.astype(np.float32))),
            "sm_root": (("time", "lat", "lon"), expand_daily(sm_root.astype(np.float32))),
            "co2_ppm": (("time", "lat", "lon"), expand_daily(co2)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


class StubYieldModel(nn.Module):
    """
    Deterministic yield: hotter / wetter sites score higher.

    Uses :data:`CLIMATE_CHANNEL_NAMES` indices (11-channel tensors).
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
        assert climate.shape[-1] == N_CLIMATE_CHANNELS
        tmax_mean = climate[..., CLIMATE_IDX["tmax"]].mean(dim=1)
        precip_sum = climate[..., CLIMATE_IDX["precip"]].sum(dim=1)
        return self.a_tmax * tmax_mean + self.b_precip * precip_sum


def test_extract_daily_climate_11ch_shape_contract() -> None:
    grid = _synthetic_climate_grids()
    arr = extract_daily_climate_11ch(grid, lat=6.0, lon=-5.0, year=2020)
    assert arr.shape == (365, N_CLIMATE_CHANNELS)
    assert arr.dtype == np.float32
    tmax_col = arr[:, CLIMATE_IDX["tmax"]]
    tmin_col = arr[:, CLIMATE_IDX["tmin"]]
    tmean_col = arr[:, CLIMATE_IDX["tmean"]]
    np.testing.assert_allclose(tmean_col, 0.5 * (tmax_col + tmin_col), rtol=1e-5)


def test_et0_recomputation_matches_attrici_fao_port() -> None:
    rng = np.random.default_rng(42)
    n = 365
    tmean = rng.uniform(18.0, 32.0, n).astype(np.float64)
    rh = rng.uniform(40.0, 95.0, n).astype(np.float64)
    wind = rng.uniform(0.5, 8.0, n).astype(np.float64)
    srad = rng.uniform(5.0, 25.0, n).astype(np.float64)

    et0_numpy = _fao_et0_numpy(tmean, rh, wind, srad)
    et0_xr = _fao_et0_array(
        xr.DataArray(tmean),
        xr.DataArray(rh),
        xr.DataArray(wind),
        xr.DataArray(srad),
    )
    np.testing.assert_allclose(et0_numpy, np.asarray(et0_xr.values), atol=1e-4)

    grid = _synthetic_climate_grids()
    stack = extract_daily_climate_11ch(grid, 6.0, -5.0, 2020)
    np.testing.assert_allclose(
        stack[:, CLIMATE_IDX["et0"]],
        _fao_et0_numpy(
            stack[:, CLIMATE_IDX["tmean"]],
            stack[:, CLIMATE_IDX["rh_mean"]],
            stack[:, CLIMATE_IDX["wind10m"]],
            stack[:, CLIMATE_IDX["srad"]],
        ),
        atol=1e-4,
    )


def test_attribution_gap_zero_when_identical_climate() -> None:
    factual = _synthetic_climate_grids()
    counterfactual = factual.copy(deep=True)

    farm_coords = pd.DataFrame(
        {
            "farm_id": ["F1"],
            "lat": [6.0],
            "lon": [-5.0],
            "year": [2020],
        }
    )
    static_features = np.zeros((1, 13), dtype=np.float32)
    model = StubYieldModel(a_tmax=0.2, b_precip=0.001)

    df = climate_attributable_loss(
        factual,
        counterfactual,
        model,
        static_features,
        farm_coords,
        n_mc_samples=10,
        device="cpu",
    )
    assert abs(float(df["climate_loss_tpha"].iloc[0])) < 1e-6


def test_climate_attributable_loss_positive_when_factual_warmer() -> None:
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
    static_features = np.zeros((2, 13), dtype=np.float32)
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


def test_extract_daily_climate_4ch_deprecated_wrapper() -> None:
    grid = _synthetic_climate_grids()
    with pytest.warns(DeprecationWarning):
        arr4 = extract_daily_climate_4ch(grid, 6.0, -5.0, 2020)
    assert arr4.shape == (365, 4)
    arr11 = extract_daily_climate_11ch(grid, 6.0, -5.0, 2020)
    for i, name in enumerate(("tmax", "tmin", "precip", "srad")):
        np.testing.assert_allclose(arr4[:, i], arr11[:, CLIMATE_IDX[name]])


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
