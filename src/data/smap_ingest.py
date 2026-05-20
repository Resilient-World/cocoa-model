"""
SMAP L4 soil moisture ingest (9 km) via Google Earth Engine + Xee.

Collection:
    ``NASA/SMAP/SPL4SMGP/008``

Variables:
    - ``sm_rootzone`` (m³/m³)
    - ``sm_surface`` (m³/m³)

This is intentionally similar to :class:`~data.era5_ingest.ERA5Ingest`:
builds a lazy xarray dataset and supports Zarr persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

SMAP_L4 = "NASA/SMAP/SPL4SMGP/008"

DEFAULT_SCALE_M = 9_000


@dataclass
class SMAPIngest:
    aoi: ee.Geometry
    start: str
    end: str
    scale: int = DEFAULT_SCALE_M
    chunks: dict[str, int] | None = None
    project: str | None = None
    _dataset: xr.Dataset | None = None

    def __post_init__(self) -> None:
        self.chunks = self.chunks or {"time": 30, "latitude": 256, "longitude": 256}

    def build(self) -> xr.Dataset:
        initialize_earth_engine(project=self.project)

        ic = (
            ee.ImageCollection(SMAP_L4)
            .filterDate(self.start, self.end)
            .filterBounds(self.aoi)
            .select(["sm_rootzone", "sm_surface"])
        )
        ds = xr.open_dataset(
            ic,
            engine="ee",
            geometry=self.aoi,
            scale=self.scale,
            chunks=self.chunks,
        )
        rename: dict[str, str] = {}
        if "lat" in ds.dims:
            rename["lat"] = "latitude"
        if "lon" in ds.dims:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)

        # Ensure float32 and standard attrs.
        for v in list(ds.data_vars):
            ds[v] = ds[v].astype(np.float32)
        ds.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection": SMAP_L4,
                "start_date": self.start,
                "end_date": self.end,
            }
        )
        self._dataset = ds
        return ds

    def to_zarr(self, path: str | Path, *, mode: str = "w") -> Path:
        ds = self._dataset or self.build()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(out_path, mode=mode, consolidated=True)
        return out_path


__all__ = ["SMAPIngest"]

