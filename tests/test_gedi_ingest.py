"""Unit tests for :mod:`data.gedi_ingest`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from data.gedi_ingest import GEDIIngest


def test_agroforestry_index_threshold_logic() -> None:
    # Build a tiny synthetic monthly time series with high variability and high canopy.
    time = xr.date_range("2023-01-01", periods=12, freq="MS")
    rh98 = xr.DataArray(
        np.array([5, 20, 6, 22, 7, 18, 8, 21, 6, 23, 7, 19], dtype=np.float32).reshape(12, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [6.0], "longitude": [-1.2]},
    )
    ds = xr.Dataset({"rh98": rh98})

    ingest = GEDIIngest(aoi=MagicMock(), start="2023-01-01", end="2023-12-31")
    ingest._dataset = ds

    idx = ingest.agroforestry_index()
    val = float(idx.values.reshape(-1)[0])
    assert val in (0.0, 1.0)
    assert val == 1.0

