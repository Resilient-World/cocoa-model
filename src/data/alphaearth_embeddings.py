"""
AlphaEarth Foundations (AEF) annual satellite embeddings via Google Earth Engine.

Pre-computed 64-dimensional embedding vectors from Google DeepMind's AlphaEarth
Foundations model (arXiv:2507.22291), exposed as Earth Engine ImageCollection
``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``.

License: CC-BY 4.0 — attribute *"AlphaEarth Foundations, Google DeepMind"*.
See ``NOTICE.md`` for commercial-use notes.
"""

from __future__ import annotations

import structlog

from pathlib import Path

import ee
import numpy as np
import xarray as xr

import xee  # noqa: F401 — registers the ``ee`` Xarray backend

from data.gee_auth import initialize_earth_engine

log = structlog.get_logger(__name__)

# Verified against Earth Engine Data Catalog (Satellite Embedding V1 Annual)
AEF_ANNUAL_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_EMBEDDING_DIM = 64
AEF_BAND_NAMES: tuple[str, ...] = tuple(f"A{i:02d}" for i in range(AEF_EMBEDDING_DIM))
AEF_SUPPORTED_YEARS: tuple[int, ...] = tuple(range(2017, 2026))
AEF_DEFAULT_SCALE_M = 10
AEF_ATTRIBUTION = "AlphaEarth Foundations, Google DeepMind"
AEF_LICENSE = "CC-BY 4.0"
AEF_ARXIV = "https://arxiv.org/abs/2507.22291"


def _normalize_aef_year(year: int) -> int:
    """Clamp calendar year to catalog coverage (2017–2025)."""
    if year in AEF_SUPPORTED_YEARS:
        return year
    if year < AEF_SUPPORTED_YEARS[0]:
        return AEF_SUPPORTED_YEARS[0]
    return AEF_SUPPORTED_YEARS[-1]


def _year_date_range(year: int) -> tuple[str, str]:
    y = _normalize_aef_year(year)
    return f"{y}-01-01", f"{y}-12-31"


class AlphaEarthIngest:
    """
    Ingest AlphaEarth annual embeddings for an AOI and calendar year.

    Each pixel is a unit-norm 64-D vector (bands ``A00``–``A63``). Use all bands
    jointly for downstream heads or similarity search.
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        year: int = 2023,
        project: str | None = None,
    ) -> None:
        self.aoi = aoi
        self.year = _normalize_aef_year(year)
        self.project = project
        self._embedding_image: ee.Image | None = None

    def _collection(self) -> ee.ImageCollection:
        start, end = _year_date_range(self.year)
        return (
            ee.ImageCollection(AEF_ANNUAL_COLLECTION)
            .filterDate(start, end)
            .filterBounds(self.aoi)
            .select(list(AEF_BAND_NAMES))
        )

    def embedding_image(self) -> ee.Image:
        """Annual embedding mosaic ``[A00 … A63]`` clipped to :attr:`aoi`."""
        if self._embedding_image is None:
            mosaic = self._collection().mosaic().select(list(AEF_BAND_NAMES))
            self._embedding_image = mosaic.clip(self.aoi)
        return self._embedding_image

    def sample_points(
        self,
        points_fc: ee.FeatureCollection,
        *,
        scale_m: int = AEF_DEFAULT_SCALE_M,
    ) -> ee.FeatureCollection:
        """
        Sample 64-band embeddings at point locations.

        Adds properties ``A00`` … ``A63`` to each feature.
        """
        initialize_earth_engine(project=self.project)
        return self.embedding_image().reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.first(),
            scale=scale_m,
        )

    def sample_point(
        self,
        lat: float,
        lon: float,
        *,
        scale_m: int = AEF_DEFAULT_SCALE_M,
    ) -> np.ndarray | None:
        """Return embedding vector ``[64]`` at a single point, or ``None`` if masked."""
        initialize_earth_engine(project=self.project)
        point = ee.Geometry.Point([float(lon), float(lat)])
        sample = self.embedding_image().reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale_m,
            bestEffort=True,
        ).getInfo() or {}

        vec = np.array(
            [float(sample.get(b, np.nan)) for b in AEF_BAND_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(vec).all():
            return None
        return vec.astype(np.float32)

    def export_to_zarr(
        self,
        out_path: str | Path,
        *,
        scale_m: int = AEF_DEFAULT_SCALE_M,
        chunks: dict[str, int] | None = None,
    ) -> Path:
        """Materialize embedding mosaic to Zarr via Xee."""
        initialize_earth_engine(project=self.project)
        chunks = chunks or {"latitude": 256, "longitude": 256}
        img = self.embedding_image()
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

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection": AEF_ANNUAL_COLLECTION,
                "year": self.year,
                "embedding_dim": AEF_EMBEDDING_DIM,
                "license": AEF_LICENSE,
                "attribution": AEF_ATTRIBUTION,
                "reference": AEF_ARXIV,
            }
        )
        ds.to_zarr(path, mode="w", consolidated=True)
        return path


__all__ = [
    "AEF_ANNUAL_COLLECTION",
    "AEF_ARXIV",
    "AEF_ATTRIBUTION",
    "AEF_BAND_NAMES",
    "AEF_DEFAULT_SCALE_M",
    "AEF_EMBEDDING_DIM",
    "AEF_LICENSE",
    "AEF_SUPPORTED_YEARS",
    "AlphaEarthIngest",
]
