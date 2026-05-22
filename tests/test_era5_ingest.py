"""Unit and integration tests for ERA5 / CHIRPS ingest."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.era5_ingest import ERA5Ingest, compute_derived_features


def _toy_dataset(n_days: int = 120) -> xr.Dataset:
    rng = np.random.default_rng(0)
    time = pd.date_range("2020-01-01", periods=n_days, freq="D")
    lat = np.array([6.0, 7.0])
    lon = np.array([-5.0, -4.0])
    shape = (n_days, len(lat), len(lon))
    return xr.Dataset(
        {
            "tmean": (("time", "lat", "lon"), 26 + rng.normal(0, 1, shape)),
            "tmax": (("time", "lat", "lon"), 31 + rng.normal(0, 1, shape)),
            "tmin": (("time", "lat", "lon"), 22 + rng.normal(0, 1, shape)),
            "rh_mean": (("time", "lat", "lon"), 80 + rng.normal(0, 3, shape).clip(-10, 10)),
            "vpd_mean": (("time", "lat", "lon"), 0.8 + rng.normal(0, 0.1, shape)),
            "precip": (("time", "lat", "lon"), np.maximum(0, rng.gamma(0.5, 4, shape))),
            "et0": (("time", "lat", "lon"), 4 + rng.normal(0, 0.3, shape)),
            "cwd": (("time", "lat", "lon"), rng.normal(0, 1, shape)),
            "sm_root": (("time", "lat", "lon"), 0.28 + rng.normal(0, 0.02, shape)),
            "wind10m": (("time", "lat", "lon"), 2 + rng.normal(0, 0.3, shape)),
            "srad": (("time", "lat", "lon"), 18 + rng.normal(0, 1, shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_magnus_es_at_25c_matches_published_value() -> None:
    # Alduchov-Eskridge 1996: es(25C) ~ 3.1690 kPa (within 0.5%)
    from data.era5_ingest import _saturation_vapor_pressure_kpa

    es = _saturation_vapor_pressure_kpa(25.0)
    assert abs(es - 3.169) / 3.169 < 0.005


def test_vpd_zero_at_full_saturation() -> None:
    from data.era5_ingest import _vpd_kpa

    assert _vpd_kpa(tmean_c=27.0, rh_pct=100.0) == pytest.approx(0.0, abs=1e-9)


def test_vpd_positive_when_rh_low() -> None:
    from data.era5_ingest import _vpd_kpa

    assert _vpd_kpa(tmean_c=30.0, rh_pct=40.0) > 1.5


def test_compute_derived_features_adds_expected_vars() -> None:
    ds = _toy_dataset()
    out = compute_derived_features(ds)
    for v in [
        "gdd_cocoa",
        "heat_days_above_32c",
        "dry_spell_max",
        "vpd_mean_30d",
        "cwd_30d",
        "sm_root_30d",
        "vpd_mean_90d",
        "cwd_90d",
        "sm_root_90d",
    ]:
        assert v in out.data_vars


def test_gdd_cocoa_caps_correctly() -> None:
    ds = _toy_dataset()
    ds["tmean"] = ds["tmean"] * 0 + 40  # uniformly hot
    out = compute_derived_features(ds)
    # cap at 32 - base 18 = 14
    assert float(out["gdd_cocoa"].max()) == pytest.approx(14.0, abs=1e-6)


def test_heat_days_count_matches_threshold() -> None:
    ds = _toy_dataset()
    ds["tmax"] = ds["tmax"] * 0 + 33
    out = compute_derived_features(ds)
    assert (
        float(out["heat_days_above_32c"].sum())
        == ds.sizes["time"] * ds.sizes["lat"] * ds.sizes["lon"]
    )


@pytest.mark.integration
def test_era5_ingest_small_aoi() -> None:
    import os

    import ee

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("No GEE credentials")

    ee.Initialize()
    aoi = ee.Geometry.Rectangle([-5.6, 6.7, -5.4, 6.9])  # tiny CDI box
    ds = ERA5Ingest(aoi, "2022-06-01", "2022-06-10").build()
    assert {"tmean", "precip", "vpd_mean", "et0", "cwd", "sm_root"} <= set(ds.data_vars)
    assert ds.sizes["time"] == 10
