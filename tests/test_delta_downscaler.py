"""Tests for 0.5° → ERA5-Land delta downscaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from counterfactual.delta_downscaler import (
    DeltaDownscaler,
    _interp_monthly_to_grid,
    _lat_lon_names,
)


def _isimip_cube(*, seed: int = 1, tas_offset: float = 0.0, pr_scale: float = 1.0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2018-01-01", periods=120, freq="D")
    lat = np.array([6.0, 7.0])
    lon = np.array([-5.0, -4.0])
    shape = (len(time), len(lat), len(lon))
    base_t = 27.0 + tas_offset
    return xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), base_t + rng.normal(0, 0.3, shape)),
            "tasmin": (("time", "lat", "lon"), base_t - 3 + rng.normal(0, 0.3, shape)),
            "tasmax": (("time", "lat", "lon"), base_t + 3 + rng.normal(0, 0.3, shape)),
            "pr": (("time", "lat", "lon"), np.maximum(0, rng.gamma(1.0, 2.0, shape) * pr_scale)),
            "hurs": (("time", "lat", "lon"), 75 + rng.normal(0, 2, shape)),
            "rsds": (("time", "lat", "lon"), 15 + rng.normal(0, 0.5, shape)),
            "sfcwind": (("time", "lat", "lon"), 2 + rng.normal(0, 0.1, shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _era5_cube() -> xr.Dataset:
    rng = np.random.default_rng(2)
    time = pd.date_range("2018-01-01", periods=120, freq="D")
    lat = np.linspace(5.95, 7.05, 5)
    lon = np.linspace(-5.05, -3.95, 5)
    shape = (len(time), len(lat), len(lon))
    return xr.Dataset(
        {
            "tmean": (("time", "latitude", "longitude"), 26 + rng.normal(0, 0.5, shape)),
            "tmax": (("time", "latitude", "longitude"), 31 + rng.normal(0, 0.5, shape)),
            "tmin": (("time", "latitude", "longitude"), 22 + rng.normal(0, 0.5, shape)),
            "rh_mean": (("time", "latitude", "longitude"), 80 + rng.normal(0, 2, shape)),
            "vpd_mean": (("time", "latitude", "longitude"), 0.8 + rng.normal(0, 0.05, shape)),
            "precip": (("time", "latitude", "longitude"), np.maximum(0, rng.gamma(0.5, 3, shape))),
            "et0": (("time", "latitude", "longitude"), 4 + rng.normal(0, 0.2, shape)),
            "cwd": (("time", "latitude", "longitude"), rng.normal(0, 0.5, shape)),
            "sm_root": (("time", "latitude", "longitude"), 0.28 + rng.normal(0, 0.01, shape)),
            "wind10m": (("time", "latitude", "longitude"), 2 + rng.normal(0, 0.1, shape)),
            "srad": (("time", "latitude", "longitude"), 18 + rng.normal(0, 0.5, shape)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_additive_delta_preserves_anomaly() -> None:
    factual_05 = _isimip_cube(seed=1, tas_offset=2.0)
    counter_05 = _isimip_cube(seed=1, tas_offset=0.0)
    era5 = _era5_cube()
    scaler = DeltaDownscaler(factual_05, counter_05)
    scaler.build_delta()
    counter_9 = scaler.apply_to_factual(era5)

    lat_name, lon_name = _lat_lon_names(era5)
    assert scaler._delta_additive is not None
    delta_9 = _interp_monthly_to_grid(scaler._delta_additive["tas"], era5, lat_name, lon_name)

    anomaly = era5["tmean"] - counter_9["tmean"]
    months = era5.time.dt.month
    expected = delta_9.sel(month=months)

    valid = np.isfinite(anomaly.values) & np.isfinite(expected.values)
    np.testing.assert_allclose(
        anomaly.values[valid],
        expected.values[valid],
        rtol=1e-4,
        atol=1e-4,
    )


def test_multiplicative_delta_nonnegative_for_pr() -> None:
    factual_05 = _isimip_cube(seed=1, pr_scale=1.2)
    counter_05 = _isimip_cube(seed=2, pr_scale=0.8)
    era5 = _era5_cube()
    scaler = DeltaDownscaler(factual_05, counter_05)
    counter_9 = scaler.apply_to_factual(era5)
    assert float(counter_9["precip"].min()) >= 0.0


def test_apply_to_factual_preserves_time_index() -> None:
    factual_05 = _isimip_cube(seed=1)
    counter_05 = _isimip_cube(seed=2)
    era5 = _era5_cube()
    scaler = DeltaDownscaler(factual_05, counter_05)
    counter_9 = scaler.apply_to_factual(era5)
    assert counter_9.indexes["time"].equals(era5.indexes["time"])
