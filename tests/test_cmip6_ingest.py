"""Unit and integration tests for CMIP6 (GDDP-CMIP6) ingest and scenario builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from counterfactual.cmip6_scenarios import ScenarioBuilder
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES


def _toy_hist() -> xr.Dataset:
    time = pd.date_range("2010-01-01", periods=60, freq="D")
    lat = np.array([6.0])
    lon = np.array([-5.0])
    shape = (len(time), len(lat), len(lon))
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "tmean": (("time", "latitude", "longitude"), 26 + rng.normal(0, 0.3, shape)),
            "tmax": (("time", "latitude", "longitude"), 31 + rng.normal(0, 0.3, shape)),
            "tmin": (("time", "latitude", "longitude"), 22 + rng.normal(0, 0.3, shape)),
            "rh_mean": (("time", "latitude", "longitude"), 80 + rng.normal(0, 2.0, shape)),
            "precip": (("time", "latitude", "longitude"), np.maximum(0, rng.gamma(0.6, 3.0, shape))),
            "srad": (("time", "latitude", "longitude"), 18 + rng.normal(0, 1.0, shape)),
            "wind10m": (("time", "latitude", "longitude"), 2 + rng.normal(0, 0.2, shape)),
            "sm_root": (("time", "latitude", "longitude"), 0.28 + rng.normal(0, 0.01, shape)),
            "vpd_mean": (("time", "latitude", "longitude"), 0.9 + rng.normal(0, 0.05, shape)),
            "et0": (("time", "latitude", "longitude"), 4.0 + rng.normal(0, 0.2, shape)),
            "cwd": (("time", "latitude", "longitude"), rng.normal(0, 1.0, shape)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds["cwd_cum"] = ds["cwd"].cumsum("time")
    return ds


def _toy_cmip6(hist: xr.Dataset) -> xr.Dataset:
    # Create a small future cube with raw vars and derived vars, with model/scenario dims.
    fut = hist.isel(time=slice(0, 20)).copy()
    fut = fut.assign_coords(time=pd.date_range("2030-01-01", periods=20, freq="D"))
    # Raw units
    raw = xr.Dataset(
        {
            "tas": fut["tmean"] + 273.15 + 1.0,
            "tasmax": fut["tmax"] + 273.15 + 1.0,
            "tasmin": fut["tmin"] + 273.15 + 1.0,
            "hurs": fut["rh_mean"] - 1.0,
            "pr": (fut["precip"] * 1.1) / 86400.0,
            "rsds": (fut["srad"] * 1.05) * 1e6 / 86400.0,
            "sfcWind": fut["wind10m"] * 1.02,
        }
    )
    raw = raw.expand_dims(model=["M"], scenario=["ssp245"])
    return raw


def test_scenario_builder_produces_pinn_ready_tensor(tmp_path: Path) -> None:
    hist = _toy_hist()
    cmip6 = _toy_cmip6(hist)

    hist_path = tmp_path / "hist.zarr"
    cmip6_path = tmp_path / "cmip6.zarr"
    hist.to_zarr(hist_path)
    cmip6.to_zarr(cmip6_path)

    sb = ScenarioBuilder(str(hist_path), str(cmip6_path))
    out = sb.build_scenario("ssp245", ("2030-01-01", "2030-01-20"))

    for name in ("tmax", "tmin", "tmean", "precip", "srad", "wind10m", "rh_mean", "et0", "cwd"):
        assert name in out.data_vars
    assert "vpd" in out.data_vars
    assert "co2_ppm" in out.data_vars
    for required in CLIMATE_CHANNEL_NAMES:
        assert required in out.data_vars


def test_delta_change_factors_preserve_monthly_climatology(tmp_path: Path) -> None:
    # If future raw is constructed as +1K shift, scenario mean should reflect that shift.
    hist = _toy_hist()
    cmip6 = _toy_cmip6(hist)
    hist_path = tmp_path / "hist.zarr"
    cmip6_path = tmp_path / "cmip6.zarr"
    hist.to_zarr(hist_path)
    cmip6.to_zarr(cmip6_path)

    sb = ScenarioBuilder(str(hist_path), str(cmip6_path))
    out = sb.build_scenario("ssp245", ("2030-01-01", "2030-01-20"))
    assert float((out["tmean"] - hist["tmean"].isel(time=slice(0, 20))).mean()) > 0.2


@pytest.mark.integration
def test_cmip6_ingest_tiny_aoi_real_gee() -> None:
    import os

    import ee

    from data.cmip6_ingest import CMIP6Ingest

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("No GEE credentials")

    ee.Initialize()
    aoi = ee.Geometry.Rectangle([-5.6, 6.7, -5.4, 6.9])
    ingest = CMIP6Ingest(
        aoi,
        "2030-06-01",
        "2030-06-05",
        models=["CESM2"],
        scenarios=["ssp245"],
    )
    ds = ingest.build_ensemble()
    assert {"tmean", "precip", "vpd_mean", "et0", "cwd"} <= set(ds.data_vars)
    assert ds.sizes["time"] == 5

