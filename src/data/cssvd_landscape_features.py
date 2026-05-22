"""
Landscape + climate covariates for Dumont et al. (2025) CSSVD incidence modeling.

GEE: ESA WorldCover 10 m buffers, CHIRPS extreme precipitation, ERA5 diurnal range.
Local: optional ERA5 Zarr cache for batch training.
"""

from __future__ import annotations

import structlog

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
import xarray as xr

from data.cocoa_exposure import (
    DEFAULT_AGRIFM_CHECKPOINT,
    DEFAULT_AEF_CHECKPOINT,
    DEFAULT_GALILEO_CHECKPOINT,
    DEFAULT_SCALE_M,
    _REPO_ROOT,
    processed_era5_zarr_path,
    region_for_point,
    sample_cocoa_probability_at_point,
)
from data.cssvd_strain_atlas import StrainRegion, lookup_strain_region
from data.era5_ingest import CHIRPS_DAILY, ERA5_DAILY
from data.gee_auth import initialize_earth_engine

log = structlog.get_logger(__name__)

ESA_WORLDCOVER = "ESA/WorldCover/v200/10m"
WORLDCOVER_BAND = "Map"
WORLDCOVER_SCALE_M = 10
GROWING_SEASON_MONTHS = (4, 5, 6, 7, 8, 9)  # Apr–Sep
EXTREME_PRECIP_MM = 100.0
EXTREME_PRECIP_WINDOW_DAYS = 5
HORIZON_MONTHS = 12.0


@dataclass(frozen=True)
class BufferComposition:
    """Land-cover composition within a circular buffer (GEE sample)."""

    cocoa_fraction: float
    non_cocoa_fraction: float
    class_fractions: dict[str, float]
    n_pixels: int


@dataclass(frozen=True)
class LandscapeFeatureRow:
    """Per-pixel feature vector for :class:`~hazards.cssvd_landscape.LandscapeCSSVDModel`."""

    lat: float
    lon: float
    year: int
    cocoa_probability_local: float
    non_cocoa_buffer_500m: float
    canopy_fragmentation_index: float
    extreme_precip_5day_count_yr: int
    dtr_growing_season: float
    strain_region: StrainRegion

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_buffer(lat: float, lon: float, buffer_m: float) -> ee.Geometry:
    return ee.Geometry.Point([float(lon), float(lat)]).buffer(float(buffer_m))


def _parse_frequency_histogram(raw: Any) -> dict[int, float]:
    """Parse GEE ``frequencyHistogram`` reducer output to class_id -> fraction."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        return {}
    total = 0.0
    counts: dict[int, float] = {}
    for key, val in items:
        try:
            cls = int(float(key))
            cnt = float(val)
        except (TypeError, ValueError):
            continue
        counts[cls] = cnt
        total += cnt
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def shannon_diversity(class_fractions: dict[int, float]) -> float:
    """Shannon entropy H = -sum p log(p), excluding zero-mass classes."""
    probs = [p for p in class_fractions.values() if p > 0]
    if not probs:
        return 0.0
    arr = np.asarray(probs, dtype=np.float64)
    return float(-np.sum(arr * np.log(arr)))


def _worldcover_image() -> ee.Image:
    return ee.ImageCollection(ESA_WORLDCOVER).first().select(WORLDCOVER_BAND)


def sample_worldcover_histogram(
    lat: float,
    lon: float,
    *,
    buffer_m: float = 500,
    project: str | None = None,
    scale_m: int = WORLDCOVER_SCALE_M,
) -> tuple[dict[int, float], int]:
    """WorldCover class fractions in a buffer (GEE ``frequencyHistogram``)."""
    initialize_earth_engine(project=project)
    geom = _point_buffer(lat, lon, buffer_m)
    wc = _worldcover_image()
    hist = wc.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geom,
        scale=scale_m,
        bestEffort=True,
        maxPixels=1e9,
    ).getInfo() or {}
    raw = hist.get(WORLDCOVER_BAND, hist.get("Map"))
    fractions = _parse_frequency_histogram(raw)
    n_pixels = int(sum(float(v) for v in (raw or {}).values()) if isinstance(raw, dict) else 0)
    return fractions, n_pixels


def sample_canopy_fragmentation_index(
    lat: float,
    lon: float,
    *,
    buffer_m: float = 1000,
    project: str | None = None,
) -> float:
    """Shannon diversity of land-cover classes in a 1 km WorldCover buffer."""
    fractions, _ = sample_worldcover_histogram(
        lat, lon, buffer_m=buffer_m, project=project
    )
    return shannon_diversity(fractions)


def _cocoa_fraction_gee(
    lat: float,
    lon: float,
    *,
    buffer_m: float,
    cocoa_prob_threshold: float,
    year: int,
    project: str | None,
) -> float:
    """Fraction of buffer pixels with ensemble P(cocoa) >= threshold (GEE reducers)."""
    initialize_earth_engine(project=project)
    geom = _point_buffer(lat, lon, buffer_m)
    prob = sample_cocoa_probability_at_point(
        lat,
        lon,
        year=year,
        backend="ensemble_v2",
        project=project,
        galileo_checkpoint=DEFAULT_GALILEO_CHECKPOINT,
        aef_checkpoint=DEFAULT_AEF_CHECKPOINT,
        agrifm_checkpoint=DEFAULT_AGRIFM_CHECKPOINT,
    )
    # Local point prob seeds a constant mosaic for buffer fraction (GEE quota friendly).
    cocoa_mask = ee.Image.constant(prob >= cocoa_prob_threshold).rename("cocoa")
    one = ee.Image.constant(1).rename("one")
    stacked = one.addBands(cocoa_mask)
    stats = stacked.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=DEFAULT_SCALE_M,
        bestEffort=True,
        maxPixels=1e9,
    ).getInfo() or {}
    total = float(stats.get("one", 0) or 0)
    cocoa_n = float(stats.get("cocoa", 0) or 0)
    if total <= 0:
        return float(prob >= cocoa_prob_threshold)
    return float(np.clip(cocoa_n / total, 0.0, 1.0))


def sample_buffer_composition(
    lat: float,
    lon: float,
    *,
    buffer_m: float = 500,
    cocoa_prob_threshold: float = 0.5,
    year: int = 2023,
    project: str | None = None,
) -> BufferComposition:
    """
    Buffer land-cover composition using ESA WorldCover + ensemble cocoa mask.

    ``non_cocoa_fraction`` proxies vector dilution (Dumont et al. 2025).
    """
    class_fracs_int, n_pixels = sample_worldcover_histogram(
        lat, lon, buffer_m=buffer_m, project=project
    )
    cocoa_fraction = _cocoa_fraction_gee(
        lat,
        lon,
        buffer_m=buffer_m,
        cocoa_prob_threshold=cocoa_prob_threshold,
        year=year,
        project=project,
    )
    non_cocoa = float(np.clip(1.0 - cocoa_fraction, 0.0, 1.0))
    class_fractions = {str(k): float(v) for k, v in class_fracs_int.items()}
    return BufferComposition(
        cocoa_fraction=cocoa_fraction,
        non_cocoa_fraction=non_cocoa,
        class_fractions=class_fractions,
        n_pixels=n_pixels,
    )


def _extreme_precip_from_zarr(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
) -> int | None:
    if "precip" not in ds:
        return None
    sub = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if sub.sizes.get("time", 0) < EXTREME_PRECIP_WINDOW_DAYS + 1:
        return None
    pt = sub.sel(latitude=lat, longitude=lon, method="nearest")
    precip = pt["precip"].astype(np.float64)
    if hasattr(precip, "compute"):
        precip = precip.compute()
    arr = np.asarray(precip.values).ravel()
    if arr.size < EXTREME_PRECIP_WINDOW_DAYS + 1:
        return None
    roll = pd.Series(arr).rolling(EXTREME_PRECIP_WINDOW_DAYS, min_periods=EXTREME_PRECIP_WINDOW_DAYS).sum()
    return int((roll > EXTREME_PRECIP_MM).sum())


def _dtr_growing_season_from_zarr(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
) -> float | None:
    if "tmax" not in ds or "tmin" not in ds:
        return None
    sub = ds.sel(time=slice(f"{year}-04-01", f"{year}-09-30"))
    if sub.sizes.get("time", 0) == 0:
        return None
    pt = sub.sel(latitude=lat, longitude=lon, method="nearest")
    dtr = (pt["tmax"] - pt["tmin"]).astype(np.float64)
    if hasattr(dtr, "compute"):
        dtr = dtr.compute()
    val = float(np.nanmean(np.asarray(dtr.values)))
    return val if np.isfinite(val) else None


def _extreme_precip_gee(
    lat: float,
    lon: float,
    year: int,
    *,
    project: str | None = None,
) -> int:
    """Count days with 5-day CHIRPS cumulative precipitation > 100 mm (server-side)."""
    initialize_earth_engine(project=project)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    geom = ee.Geometry.Point([float(lon), float(lat)])
    coll = (
        ee.ImageCollection(CHIRPS_DAILY)
        .filterDate(start, end)
        .filterBounds(geom)
        .select("precipitation")
    )

    def _rolling5(img: ee.Image) -> ee.Image:
        millis = img.date().millis()
        window = coll.filterDate(
            img.date().advance(-4, "day"),
            img.date().advance(1, "day"),
        )
        roll5 = window.sum().rename("roll5")
        return roll5.set("system:time_start", millis)

    rolled = coll.map(_rolling5)
    extreme = rolled.map(lambda im: im.gt(EXTREME_PRECIP_MM).rename("extreme"))
    count_img = extreme.sum()
    sample = count_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=geom,
        scale=5000,
        bestEffort=True,
    ).getInfo() or {}
    raw = sample.get("extreme", 0)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _dtr_growing_season_gee(
    lat: float,
    lon: float,
    year: int,
    *,
    project: str | None = None,
) -> float:
    """Mean diurnal temperature range (tmax - tmin) for Apr–Sep (ERA5 daily)."""
    initialize_earth_engine(project=project)
    geom = ee.Geometry.Point([float(lon), float(lat)])
    start = f"{year}-04-01"
    end = f"{year}-09-30"
    era5 = (
        ee.ImageCollection(ERA5_DAILY)
        .filterDate(start, end)
        .filterBounds(geom)
        .select(["maximum_2m_air_temperature", "minimum_2m_air_temperature"])
    )

    def _dtr(img: ee.Image) -> ee.Image:
        tmax = img.select("maximum_2m_air_temperature").subtract(273.15)
        tmin = img.select("minimum_2m_air_temperature").subtract(273.15)
        return tmax.subtract(tmin).rename("dtr")

    mean_dtr = era5.map(_dtr).mean()
    sample = mean_dtr.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=geom,
        scale=11_000,
        bestEffort=True,
    ).getInfo() or {}
    raw = sample.get("dtr", 0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _open_era5_zarr_for_point(lat: float, lon: float) -> xr.Dataset | None:
    region = region_for_point(lat, lon)
    if region is None:
        return None
    path = processed_era5_zarr_path(region)
    if not path.is_dir():
        path = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
    if not path.is_dir():
        return None
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception as exc:
        log.debug("ERA5 Zarr open failed: %s", exc)
        return None


def build_landscape_feature_row(
    lat: float,
    lon: float,
    year: int,
    *,
    project: str | None = None,
    cocoa_prob_threshold: float = 0.5,
    use_gee_climate: bool = False,
) -> LandscapeFeatureRow:
    """
    Assemble all landscape/climate covariates for one farm pixel.
    """
    cocoa_p = sample_cocoa_probability_at_point(
        lat,
        lon,
        year=year,
        backend="ensemble_v2",
        project=project,
        galileo_checkpoint=DEFAULT_GALILEO_CHECKPOINT,
        aef_checkpoint=DEFAULT_AEF_CHECKPOINT,
        agrifm_checkpoint=DEFAULT_AGRIFM_CHECKPOINT,
    )
    buffer = sample_buffer_composition(
        lat,
        lon,
        buffer_m=500,
        cocoa_prob_threshold=cocoa_prob_threshold,
        year=year,
        project=project,
    )
    fragmentation = sample_canopy_fragmentation_index(
        lat, lon, buffer_m=1000, project=project
    )
    strain = lookup_strain_region(lat, lon)

    extreme_count: int | None = None
    dtr: float | None = None
    if not use_gee_climate:
        ds = _open_era5_zarr_for_point(lat, lon)
        if ds is not None:
            try:
                extreme_count = _extreme_precip_from_zarr(ds, lat, lon, year)
                dtr = _dtr_growing_season_from_zarr(ds, lat, lon, year)
            finally:
                ds.close()

    if extreme_count is None or use_gee_climate:
        extreme_count = _extreme_precip_gee(lat, lon, year, project=project)
    if dtr is None or use_gee_climate:
        dtr = _dtr_growing_season_gee(lat, lon, year, project=project)

    return LandscapeFeatureRow(
        lat=float(lat),
        lon=float(lon),
        year=int(year),
        cocoa_probability_local=float(cocoa_p),
        non_cocoa_buffer_500m=float(buffer.non_cocoa_fraction),
        canopy_fragmentation_index=float(fragmentation),
        extreme_precip_5day_count_yr=int(extreme_count),
        dtr_growing_season=float(dtr),
        strain_region=strain,
    )


def landscape_features_cache_path(year: int, *, repo_root: Path | None = None) -> Path:
    root = repo_root or _REPO_ROOT
    return root / "data" / "processed" / f"cssvd_landscape_features_{year}.parquet"


__all__ = [
    "BufferComposition",
    "ESA_WORLDCOVER",
    "HORIZON_MONTHS",
    "LandscapeFeatureRow",
    "build_landscape_feature_row",
    "landscape_features_cache_path",
    "sample_buffer_composition",
    "sample_canopy_fragmentation_index",
    "sample_worldcover_histogram",
    "shannon_diversity",
]
