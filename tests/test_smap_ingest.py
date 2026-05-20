"""Unit tests for :mod:`data.smap_ingest`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr

from data.smap_ingest import SMAPIngest


def test_smap_to_zarr_writes_expected_schema(tmp_path: Path) -> None:
    time = pd.date_range("2023-01-01", periods=10, freq="D")
    lat = np.array([6.0], dtype=np.float32)
    lon = np.array([-1.2], dtype=np.float32)
    shape = (len(time), 1, 1)
    ds = xr.Dataset(
        {
            "sm_rootzone": (("time", "latitude", "longitude"), np.full(shape, 0.25, dtype=np.float32)),
            "sm_surface": (("time", "latitude", "longitude"), np.full(shape, 0.18, dtype=np.float32)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )

    ingest = SMAPIngest(aoi=MagicMock(), start="2023-01-01", end="2023-01-10")
    ingest._dataset = ds
    out = ingest.to_zarr(tmp_path / "smap.zarr")

    reopened = xr.open_zarr(out, consolidated=True)
    assert set(reopened.data_vars) == {"sm_rootzone", "sm_surface"}

