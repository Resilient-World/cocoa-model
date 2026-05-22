"""
GEDI canopy structure ingest (monthly products).

L2A:
    ``LARSE/GEDI/GEDI02_A_002_MONTHLY`` — waveform metrics (e.g. rh98)
L4A (optional):
    ``LARSE/GEDI/GEDI04_A_002_MONTHLY`` — aboveground biomass density (AGBD)

This module provides:
- A lazy Xarray dataset over an AOI and time range
- Summary fields for canopy height + AGBD
- A simple agroforestry/shade proxy derived from GEDI variability and height
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

GEDI_L2A_MONTHLY = "LARSE/GEDI/GEDI02_A_002_MONTHLY"
GEDI_L4A_MONTHLY = "LARSE/GEDI/GEDI04_A_002_MONTHLY"

DEFAULT_SCALE_M = 1000  # GEDI gridded monthly products are ~1 km


@dataclass
class GEDIIngest:
    aoi: ee.Geometry
    start: str
    end: str
    include_agbd: bool = True
    rh_band: str = "rh98"
    scale: int = DEFAULT_SCALE_M
    chunks: dict[str, int] | None = None
    project: str | None = None
    _dataset: xr.Dataset | None = None

    def __post_init__(self) -> None:
        self.chunks = self.chunks or {"time": 12, "latitude": 256, "longitude": 256}

    def build(self) -> xr.Dataset:
        initialize_earth_engine(project=self.project)

        l2a = (
            ee.ImageCollection(GEDI_L2A_MONTHLY)
            .filterDate(self.start, self.end)
            .filterBounds(self.aoi)
            .select([self.rh_band])
        )

        ic = l2a
        if self.include_agbd:
            l4a = (
                ee.ImageCollection(GEDI_L4A_MONTHLY)
                .filterDate(self.start, self.end)
                .filterBounds(self.aoi)
                .select(["agbd"])
            )

            def _join(img: ee.Image) -> ee.Image:
                millis = img.date().millis()
                agbd_img = l4a.filter(ee.Filter.eq("system:time_start", millis)).first()
                return cast(
                    ee.Image,
                    ee.Image.cat([img, ee.Image(agbd_img)]).copyProperties(
                        img, ["system:time_start"]
                    ),
                )

            ic = l2a.map(_join)

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

        # Normalize variable names.
        if self.rh_band in ds.data_vars:
            ds = ds.rename({self.rh_band: "rh98"})

        self._dataset = ds
        return ds

    def summary(self) -> xr.Dataset:
        """
        Reduce monthly GEDI to canopy/biomass summary fields.

        Outputs:
        - canopy_height_mean: mean rh98 over time (m)
        - canopy_height_p95: 95th percentile rh98 over time (m)
        - canopy_height_cv: coefficient of variation of rh98
        - agbd_mean: mean AGBD (Mg/ha), when available
        """
        ds = self._dataset or self.build()

        rh = ds["rh98"]
        canopy_mean = rh.mean(dim="time", skipna=True).rename("canopy_height_mean")
        canopy_p95 = rh.quantile(0.95, dim="time", skipna=True).rename("canopy_height_p95")
        canopy_std = rh.std(dim="time", skipna=True)
        canopy_cv = (canopy_std / canopy_mean.where(canopy_mean != 0)).rename("rh98_cv")

        out = xr.Dataset(
            {
                "canopy_height_mean": canopy_mean.astype(np.float32),
                "canopy_height_p95": canopy_p95.astype(np.float32),
                "rh98_cv": canopy_cv.astype(np.float32),
            }
        )
        if "agbd" in ds.data_vars:
            out["agbd_mean"] = ds["agbd"].mean(dim="time", skipna=True).astype(np.float32)
        else:
            out["agbd_mean"] = xr.full_like(canopy_mean, np.nan).astype(np.float32)

        out.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection_l2a": GEDI_L2A_MONTHLY,
                "collection_l4a": GEDI_L4A_MONTHLY if self.include_agbd else None,
                "start_date": self.start,
                "end_date": self.end,
            }
        )
        return out

    def agroforestry_index(self) -> xr.DataArray:
        """
        Heuristic shade-canopy presence indicator.

        Definition (per request):
        - canopy_height_p95 > 8 m
        - rh98_cv > 0.3
        """
        s = self.summary()
        idx = ((s["canopy_height_p95"] > 8.0) & (s["rh98_cv"] > 0.3)).astype(np.float32)
        idx = idx.rename("agroforestry_index")
        idx.attrs.update({"definition": "1[canopy_height_p95>8m & rh98_cv>0.3]"})
        return idx


__all__ = ["GEDIIngest"]
