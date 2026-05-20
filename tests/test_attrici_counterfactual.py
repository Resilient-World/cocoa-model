"""Tests for ATTRICI-style counterfactual climate (Mengel et al. 2021)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.attrici_counterfactual import (
    ATTRICICounterfactual,
    hazard_return_period_shift,
    recompute_derived_counterfactuals,
)


def _synthetic_warming_dataset(n_years: int = 40, seed: int = 0) -> tuple[xr.Dataset, pd.Series]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("1980-01-01", periods=n_years * 365, freq="D")
    years = days.year.to_numpy()
    gmt_annual = pd.Series(
        np.linspace(0.0, 1.5, n_years),
        index=range(1980, 1980 + n_years),
        name="gmt",
    )
    gmt_daily = gmt_annual.reindex(years).to_numpy()
    doy = days.dayofyear.to_numpy()
    seasonal = 26 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
    tmax = seasonal + 1.2 * gmt_daily + rng.normal(0, 1.0, len(days))
    precip = np.maximum(0, rng.gamma(0.5, 4, len(days)) - 0.2 * gmt_daily)
    ds = xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), tmax[:, None, None]),
            "precip": (("time", "lat", "lon"), precip[:, None, None]),
        },
        coords={"time": days, "lat": [6.5], "lon": [-1.2]},
    )
    return ds, gmt_annual


def test_counterfactual_removes_trend_in_tmax() -> None:
    ds, gmt = _synthetic_warming_dataset()
    cf = ATTRICICounterfactual(gmt, variables=("tmax", "precip"), mode="fast").fit_transform(ds)
    fac_trend = np.polyfit(np.arange(len(ds.time)), ds["tmax"].values.ravel(), 1)[0]
    cf_trend = np.polyfit(np.arange(len(cf.time)), cf["tmax_cf"].values.ravel(), 1)[0]
    assert abs(cf_trend) < 0.3 * abs(fac_trend)


def test_precip_counterfactual_nonnegative() -> None:
    ds, gmt = _synthetic_warming_dataset()
    cf = ATTRICICounterfactual(gmt, variables=("tmax", "precip"), mode="fast").fit_transform(ds)
    assert float(cf["precip_cf"].min()) >= 0.0


def test_return_period_shift_detects_warming() -> None:
    ds, gmt = _synthetic_warming_dataset()
    cf = ATTRICICounterfactual(gmt, variables=("tmax",), mode="fast").fit_transform(ds)
    combined = xr.merge([ds, cf])
    out = hazard_return_period_shift(combined, variable="tmax", threshold=32.0, direction="above")
    # Factual climate should have higher exceedance probability than counterfactual
    assert float(out["p_factual"].mean()) > float(out["p_counterfactual"].mean())
    assert float(out["far"].mean()) > 0.0


def test_recompute_derived_skips_when_inputs_missing() -> None:
    ds, gmt = _synthetic_warming_dataset()
    cf = ATTRICICounterfactual(gmt, variables=("tmax",), mode="fast").fit_transform(ds)
    out = recompute_derived_counterfactuals(cf)
    # No tmin/rh/srad/wind -> derived vars should not be added
    assert "et0_cf" not in out.data_vars
    assert "vpd_mean_cf" not in out.data_vars


def test_bayesian_mode_requires_attrici_or_raises() -> None:
    ds, gmt = _synthetic_warming_dataset(n_years=5)
    try:
        import attrici  # noqa: F401

        pytest.skip("attrici installed; bayesian path executes")
    except ImportError:
        with pytest.raises(NotImplementedError):
            ATTRICICounterfactual(gmt, mode="bayesian").fit_transform(ds)


@pytest.mark.integration
def test_load_gistemp_loti_returns_smoothed_series() -> None:
    from data.attrici_counterfactual import load_gistemp_loti

    s = load_gistemp_loti(start_year=1950, smooth_window=21)
    assert s.index.min() >= 1950
    assert s.is_monotonic_increasing or s.diff().abs().mean() < 0.1
