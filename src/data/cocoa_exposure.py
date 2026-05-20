"""
Forest Data Partnership (FDP) cocoa probability ingest via Google Earth Engine.

Dataset
-------
Forest Data Partnership (2025) *Cocoa Probability model 2025a*, successor to
Kalischek et al. (2023) *Nature Food*. ImageCollection:
``projects/forestdatapartnership/assets/cocoa/model_2025a`` — 10 m, annual composites
for 2020 and 2023; coverage Côte d'Ivoire, Ghana, Indonesia, Ecuador, Peru, Colombia.

Licensing
---------
Non-commercial Earth Engine use: CC-BY 4.0 NC — attribution required
("Produced by Google for the Forest Data Partnership"). **Commercial deployments**
must accept the Forest Data Partnership Datasets Commercial Terms of Use.

All access is server-side GEE; raw tiles are not downloaded except via optional
lazy Xarray/Zarr materialization (Xee).
"""

from __future__ import annotations

from pathlib import Path

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

FDP_COCOA_COLLECTION = "projects/forestdatapartnership/assets/cocoa/model_2025a"
PROBABILITY_BAND = "probability"
SUPPORTED_YEARS: tuple[int, ...] = (2020, 2023)
DEFAULT_THRESHOLD = 0.65
DEFAULT_SCALE_M = 10


def _normalize_year(year: int) -> int:
    """Map arbitrary calendar year to nearest supported FDP composite (2020 or 2023)."""
    if year in SUPPORTED_YEARS:
        return year
    return 2023 if year >= 2022 else 2020


def _year_date_range(year: int) -> tuple[str, str]:
    y = _normalize_year(year)
    return f"{y}-01-01", f"{y}-12-31"


class CocoaExposureIngest:
    """
    Ingest FDP cocoa probability for an AOI and calendar year.

    Default threshold 0.65 matches Kalischek et al. (2023) F1-optimal operating point
    carried forward in the FDP 2025a product documentation.
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        year: int = 2023,
        threshold: float = DEFAULT_THRESHOLD,
        project: str | None = None,
    ) -> None:
        self.aoi = aoi
        self.year = _normalize_year(year)
        self.threshold = threshold
        self.project = project
        self._probability_image: ee.Image | None = None

    def _collection(self) -> ee.ImageCollection:
        start, end = _year_date_range(self.year)
        return (
            ee.ImageCollection(FDP_COCOA_COLLECTION)
            .filterDate(start, end)
            .filterBounds(self.aoi)
            .select([PROBABILITY_BAND])
        )

    def probability_image(self) -> ee.Image:
        """Mosaic probability raster for :attr:`year` (band ``probability``, 0–1)."""
        if self._probability_image is None:
            mosaic = self._collection().mosaic().select(PROBABILITY_BAND)
            self._probability_image = mosaic.clip(self.aoi).rename("probability")
        return self._probability_image

    def binary_mask(self) -> ee.Image:
        """Binary cocoa mask where probability >= :attr:`threshold`."""
        return self.probability_image().gte(self.threshold).rename("cocoa_mask")

    def sample_point(self, lat: float, lon: float, scale_m: int = DEFAULT_SCALE_M) -> float | None:
        """
        Sample probability at a point.

        Returns ``None`` when the pixel is masked or outside FDP coverage (caller should
        fall back to a belt heuristic).
        """
        initialize_earth_engine(project=self.project)
        point = ee.Geometry.Point([lon, lat])
        img = self.probability_image()
        sample = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale_m,
            bestEffort=True,
        ).getInfo() or {}

        raw = sample.get("probability", sample.get(PROBABILITY_BAND))
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return float(np.clip(val, 0.0, 1.0))

    def to_zarr(
        self,
        path: str,
        scale_m: int = DEFAULT_SCALE_M,
        chunks: dict[str, int] | None = None,
    ) -> None:
        """Materialize the probability mosaic to Zarr via Xee (lazy until compute)."""
        initialize_earth_engine(project=self.project)
        chunks = chunks or {"latitude": 256, "longitude": 256}
        img = self.probability_image()
        ds = xr.open_dataset(
            img,
            engine="ee",
            geometry=self.aoi,
            scale=scale_m,
            chunks=chunks,
        )
        rename: dict[str, str] = {}
        if "lat" in ds.dims:
            rename["lat"] = "latitude"
        if "lon" in ds.dims:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection": FDP_COCOA_COLLECTION,
                "year": self.year,
                "threshold": self.threshold,
                "license_note": "CC-BY-4.0-NC; commercial use requires FDP Commercial Terms",
            }
        )
        ds.to_zarr(out_path, mode="w", consolidated=True)

    def area_hectares(self, region: ee.Geometry | None = None, scale_m: int = DEFAULT_SCALE_M) -> float:
        """Sum binary-mask area (ha) over ``region`` or :attr:`aoi`."""
        initialize_earth_engine(project=self.project)
        geom = region or self.aoi
        area_img = self.binary_mask().multiply(ee.Image.pixelArea())
        result = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e13,
            bestEffort=True,
        ).getInfo() or {}
        m2 = float(result.get("cocoa_mask", 0.0) or 0.0)
        return m2 / 10_000.0


__all__ = ["CocoaExposureIngest", "FDP_COCOA_COLLECTION", "DEFAULT_THRESHOLD", "SUPPORTED_YEARS"]
