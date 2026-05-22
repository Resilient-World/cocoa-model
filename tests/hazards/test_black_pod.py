"""Tests for biotic hazard models (black pod, CSSVD, mirids, composite)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hazards import (
    BlackPodRiskModel,
    CSSVDRiskModel,
    ShadeSpecies,
    apply_biotic_losses,
)
from hazards.black_pod import SHADE_BLACK_POD_MULTIPLIERS


def _daily_climate(
    *,
    n: int = 120,
    rh: float = 80.0,
    tmean: float = 24.0,
    precip: float = 10.0,
) -> xr.Dataset:
    time = pd.date_range("2023-01-01", periods=n, freq="D")
    return xr.Dataset(
        {
            "rh_mean": ("time", np.full(n, rh, dtype=np.float32)),
            "tmean": ("time", np.full(n, tmean, dtype=np.float32)),
            "precip": ("time", np.full(n, precip, dtype=np.float32)),
        },
        coords={"time": time},
    )


def test_high_humidity_high_rainfall_triggers_pressure():
    ds = _daily_climate(rh=88.0, tmean=24.0, precip=12.0)
    model = BlackPodRiskModel()
    pressure = model.seasonal_pressure(ds)
    assert pressure > 5.0


def test_shade_modifier_khaya_reduces_loss():
    ds = _daily_climate(rh=88.0, precip=12.0)
    model = BlackPodRiskModel()
    loss_unshaded = float(
        model.seasonal_yield_loss_fraction(ds, shade_species=ShadeSpecies.UNSHADED).values
    )
    loss_khaya = float(
        model.seasonal_yield_loss_fraction(ds, shade_species=ShadeSpecies.KHAYA_IVORENSIS).values
    )
    assert loss_khaya < loss_unshaded


def test_loss_fraction_caps_at_90pct():
    ds = _daily_climate(n=365, rh=95.0, tmean=25.0, precip=20.0)
    model = BlackPodRiskModel(pressure_threshold=1.0, pressure_scale=0.5)
    loss = float(model.seasonal_yield_loss_fraction(ds).values)
    assert loss <= 0.90 + 1e-6


def test_cssvd_tolerant_clone_reduces_loss():
    model = CSSVDRiskModel()
    base = model.annual_yield_loss_fraction(50.0, tolerance=1.0)
    tolerant = model.annual_yield_loss_fraction(50.0, tolerance=0.3)
    assert tolerant < base
    assert tolerant == pytest.approx(0.2117 * 0.5 * 0.3, rel=1e-3)


def test_composite_attribution_sums_to_total_loss():
    # Mild climate keeps combined loss below the 70% cap so multiplicative math is exact.
    ds = _daily_climate(rh=78.0, precip=6.0, tmean=24.0, n=60)
    climate_yield = 2.0
    out = apply_biotic_losses(
        climate_yield,
        ds,
        {
            "cssvd_prevalence_pct": 10.0,
            "cssvd_tolerance": 1.0,
            "shade_species": ShadeSpecies.UNSHADED,
        },
    )
    attr = out["loss_attribution"]
    uncapped_surv = (1.0 - attr["black_pod"]) * (1.0 - attr["cssvd"]) * (1.0 - attr["mirids"])
    assert out["surviving_fraction"] == pytest.approx(max(uncapped_surv, 0.30), rel=1e-5)
    expected_total = 1.0 - out["surviving_fraction"]
    assert out["total_loss_fraction"] == pytest.approx(expected_total, rel=1e-5)
    assert out["final_yield"] == pytest.approx(climate_yield * out["surviving_fraction"], rel=1e-5)


def test_cola_nitida_worse_than_unshaded_multiplier():
    assert (
        SHADE_BLACK_POD_MULTIPLIERS[ShadeSpecies.COLA_NITIDA]
        > SHADE_BLACK_POD_MULTIPLIERS[ShadeSpecies.UNSHADED]
    )
