"""Unit tests for :mod:`api.feature_resolver`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from api.feature_resolver import (
    FarmFeatureResolver,
    FeatureResolverConfig,
    _awc_mm_from_texture,
    _climate_tensor_from_dataset,
)


def test_awc_from_texture_in_reasonable_range() -> None:
    awc = _awc_mm_from_texture(40.0, 25.0)
    assert 40.0 <= awc <= 280.0


def test_climate_tensor_from_zarr_point(tmp_path: Path) -> None:
    time = pd.date_range("2023-01-01", periods=365, freq="D")
    lat = np.array([6.0, 7.0])
    lon = np.array([-2.0, -1.0])
    tmax = np.random.default_rng(0).normal(28, 2, (365, 2, 2)).astype(np.float32)

    ds = xr.Dataset(
        {
            "tmax": (("time", "latitude", "longitude"), tmax),
            "tmin": (("time", "latitude", "longitude"), tmax - 6),
            "tmean": (("time", "latitude", "longitude"), tmax - 3),
            "precip": (("time", "latitude", "longitude"), np.abs(tmax) * 0.1),
            "srad": (("time", "latitude", "longitude"), np.full_like(tmax, 15.0)),
            "vpd_mean": (("time", "latitude", "longitude"), np.full_like(tmax, 1.0)),
            "et0": (("time", "latitude", "longitude"), np.full_like(tmax, 3.0)),
            "sm_root": (("time", "latitude", "longitude"), np.full_like(tmax, 0.25)),
            "wind10m": (("time", "latitude", "longitude"), np.full_like(tmax, 2.0)),
            "rh_mean": (("time", "latitude", "longitude"), np.full_like(tmax, 75.0)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    zarr_path = tmp_path / "era5.zarr"
    ds.to_zarr(zarr_path, mode="w")

    resolver = FarmFeatureResolver(
        FeatureResolverConfig(
            era5_zarr_path=zarr_path,
            static_zarr_path=tmp_path / "missing_static.zarr",
            cache_dir=tmp_path / "cache",
        )
    )
    tensor = resolver.resolve_climate(6.5, -1.5, 2023)
    assert tensor.shape == (1, 365, 11)

    # Second call hits diskcache
    tensor2 = resolver.resolve_climate(6.5, -1.5, 2023)
    assert torch.allclose(tensor, tensor2)


def test_climate_tensor_channel_order() -> None:
    time = pd.date_range("2023-01-01", periods=365, freq="D")
    ds = xr.Dataset(
        {"tmax": ("time", np.linspace(20, 30, 365))},
        coords={"time": time},
    )
    ds["tmin"] = ds["tmax"] - 5
    ds["tmean"] = ds["tmax"] - 2.5
    for name in ("precip", "srad", "vpd_mean", "et0", "sm_root", "wind10m", "rh_mean"):
        ds[name] = xr.zeros_like(ds["tmax"])
    arr = _climate_tensor_from_dataset(ds, 2023)
    assert arr.shape == (365, 11)
    assert arr[:, 0].mean() == pytest.approx(25.0, rel=0.05)
