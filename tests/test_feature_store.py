"""Unit tests for :mod:`data.feature_store`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.feature_store import FeatureStore, FeatureStorePaths


def _write_era5_stub(path: Path) -> None:
    time = pd.date_range("2023-01-01", periods=365, freq="D")
    lat = np.array([6.0], dtype=np.float32)
    lon = np.array([-1.2], dtype=np.float32)
    shape = (len(time), 1, 1)
    ds = xr.Dataset(
        {
            "tmax": (("time", "latitude", "longitude"), np.full(shape, 30.0, dtype=np.float32)),
            "tmin": (("time", "latitude", "longitude"), np.full(shape, 23.0, dtype=np.float32)),
            "tmean": (("time", "latitude", "longitude"), np.full(shape, 26.5, dtype=np.float32)),
            "precip": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype=np.float32)),
            "srad": (("time", "latitude", "longitude"), np.full(shape, 15.0, dtype=np.float32)),
            "vpd_mean": (("time", "latitude", "longitude"), np.full(shape, 1.2, dtype=np.float32)),
            "et0": (("time", "latitude", "longitude"), np.full(shape, 3.5, dtype=np.float32)),
            "sm_root": (("time", "latitude", "longitude"), np.full(shape, 0.28, dtype=np.float32)),
            "wind10m": (("time", "latitude", "longitude"), np.full(shape, 2.0, dtype=np.float32)),
            "rh_mean": (("time", "latitude", "longitude"), np.full(shape, 75.0, dtype=np.float32)),
            "co2_ppm": (("time", "latitude", "longitude"), np.full(shape, 415.0, dtype=np.float32)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_zarr(path, mode="w", consolidated=True)


def test_feature_store_to_pinn_tensors_synthetic_join(tmp_path: Path) -> None:
    era5 = tmp_path / "era5.zarr"
    _write_era5_stub(era5)

    gedi = tmp_path / "gedi.zarr"
    xr.Dataset(
        {
            "canopy_height_p95": (("latitude", "longitude"), np.array([[12.0]], dtype=np.float32)),
        },
        coords={
            "latitude": np.array([6.0], dtype=np.float32),
            "longitude": np.array([-1.2], dtype=np.float32),
        },
    ).to_zarr(gedi, mode="w", consolidated=True)

    soil = tmp_path / "soil.zarr"
    xr.Dataset(
        {
            "awc_mm": (("latitude", "longitude"), np.array([[160.0]], dtype=np.float32)),
        },
        coords={
            "latitude": np.array([6.0], dtype=np.float32),
            "longitude": np.array([-1.2], dtype=np.float32),
        },
    ).to_zarr(soil, mode="w", consolidated=True)

    store = FeatureStore(
        tmp_path,
        paths=FeatureStorePaths(era5_zarr=era5, soil_zarr=soil, gedi_zarr=gedi, smap_zarr=None),
    )
    climate, static = store.to_pinn_tensors(
        locations=[(6.5, -1.25), (6.1, -1.15)],
        year=2023,
        static_feature_names=("awc_mm", "canopy_height_p95"),
    )
    assert climate.shape == (2, 365, 11)
    assert static.shape == (2, 2)
    assert float(static[0, 0].item()) == 160.0
    assert float(static[0, 1].item()) == 12.0
