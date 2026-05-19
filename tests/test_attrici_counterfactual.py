"""Tests for ISIMIP3a delta counterfactual adjustment."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.attrici_counterfactual import compute_counterfactual_delta


def _toy_grid(name: str, value: float, n_days: int = 30) -> xr.Dataset:
    time = pd.date_range("2010-01-01", periods=n_days, freq="D")
    lat = np.array([6.0, 6.5, 7.0])
    lon = np.array([-5.0, -4.5, -4.0])
    shape = (n_days, len(lat), len(lon))
    return xr.Dataset(
        {name: (("time", "lat", "lon"), np.full(shape, value))},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_delta_zero_when_obsclim_equals_counterclim() -> None:
    factual = _toy_grid("tas", 27.0)
    obsclim = _toy_grid("tas", 27.0)
    counterclim = _toy_grid("tas", 27.0)
    cf = compute_counterfactual_delta(factual, obsclim, counterclim, skip_regrid=True)
    np.testing.assert_allclose(cf["tas"].values, factual["tas"].values)


def test_delta_subtracts_warming_signal() -> None:
    # Factual ERA5 is 28C; ISIMIP obsclim 27C, counterclim 26C.
    # Warming signal = 1C -> counterfactual should be 27C.
    factual = _toy_grid("tas", 28.0)
    obsclim = _toy_grid("tas", 27.0)
    counterclim = _toy_grid("tas", 26.0)
    cf = compute_counterfactual_delta(factual, obsclim, counterclim, skip_regrid=True)
    np.testing.assert_allclose(cf["tas"].values, 27.0)


def test_precip_delta_preserves_nonnegative() -> None:
    factual = _toy_grid("pr", 5.0)
    obsclim = _toy_grid("pr", 3.0)
    counterclim = _toy_grid("pr", 4.0)
    cf = compute_counterfactual_delta(
        factual,
        obsclim,
        counterclim,
        skip_regrid=True,
        clip_precip=True,
    )
    assert (cf["pr"].values >= 0).all()


@pytest.mark.integration
def test_isimip_download_small_region() -> None:
    if not os.getenv("ISIMIP_INTEGRATION"):
        pytest.skip("set ISIMIP_INTEGRATION=1 to run network test")

    from data.attrici_counterfactual import ISIMIPCounterfactualLoader

    loader = ISIMIPCounterfactualLoader(
        bbox=(-5.6, 6.7, -5.4, 6.9),
        start="2015-01-01",
        end="2015-01-31",
        variables=["tas"],
    )
    obs = loader.load("obsclim")
    cf = loader.load("counterclim")
    assert "tas" in obs.data_vars and "tas" in cf.data_vars
    assert obs.sizes["time"] >= 28
