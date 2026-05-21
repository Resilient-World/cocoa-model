"""
Resolve farm-level climate, static site, and optional Galileo features for API inference.

Climate: ``ERA5_ZARR_PATH`` daily stack ``[365, 11]`` (or ``features_cache.zarr``).

Static (resolved source indices 0–9, mapped into the 13-d site vector for
:class:`~models.yield_surrogate.YieldSurrogateModel`):

- 0–4: SoilGrids 2.0 clay %, sand %, SOC (g/kg), CEC (cmol/kg), pH (ISRIC / GEE)
- 5: SRTM elevation (m)
- 6: slope (degrees, from SRTM)
- 7: CHIRPS long-term mean annual precipitation (mm)
- 8: distance to nearest protected area (WDPA, km)
- 9: FDP / AEF cocoa probability

LRU cache keys use ``(lat, lon, year)`` rounded to 0.05°. Set ``USE_REAL_FEATURES=false``
to use :mod:`api.geo_mock` (tests only).
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ee
import numpy as np
import torch
import xarray as xr
import xee  # noqa: F401 — registers the ``ee`` Xarray backend
from torch import Tensor

from data.cocoa_exposure import (
    DEFAULT_AGRIFM_CHECKPOINT,
    DEFAULT_THRESHOLD,
    ExposureBackend,
    FDP_COCOA_COLLECTION,
    sample_cocoa_probability_at_point,
)
from data.ensemble_weights import DEFAULT_ENSEMBLE_WEIGHTS_PATH
from data.era5_ingest import ERA5Ingest
from data.feature_store import FeatureStore
from data.gee_auth import initialize_earth_engine
from models.yield_surrogate import (
    CLIMATE_CHANNEL_NAMES,
    DEFAULT_PLANTING_DENSITY,
    DEFAULT_TREE_AGE_YEARS,
    N_STATIC_SITE,
    pack_tree_age_static,
)

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 365
SITE_STATIC_DIM = N_STATIC_SITE
DEFAULT_AWC_MM = 150.0
FARM_REGISTRY_MAX_DIST_DEG2 = 4.0
DEFAULT_CO2_PPM = 415.0
GRID_ROUND_STEP = 0.05
MEMORY_CACHE_MAX = 4096

ISRIC_SOILGRIDS_DATA_URL = "https://files.isric.org/soilgrids/latest/data/"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA5_ZARR = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
DEFAULT_STATIC_ZARR = _REPO_ROOT / "data" / "processed" / "site_static.zarr"
DEFAULT_FEATURES_CACHE_ZARR = _REPO_ROOT / "data" / "processed" / "features_cache.zarr"
DEFAULT_FARM_REGISTRY = _REPO_ROOT / "data" / "processed" / "farm_registry.parquet"
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "cache" / "api_features"

SOILGRIDS_IMAGE = "projects/soilgrids-isric/soilgrids_world"
HANSEN_IMAGE = "UMD/hansen/global_forest_change_2023_v1_11"
SRTM_IMAGE = "USGS/SRTMGL1_003"
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
WDPA_IMAGE = "WCMC/WDPA/WDPA_current/0"

# Resolved static fields (``features_cache.zarr`` / GEE sampling)
RESOLVED_STATIC_NAMES: tuple[str, ...] = (
    "clay_pct",
    "sand_pct",
    "soc_gkg",
    "cec_cmolkg",
    "ph",
    "elevation_m",
    "slope_deg",
    "chirps_annual_mm",
    "protected_dist_km",
    "cocoa_prob",
)
N_RESOLVED_STATIC = len(RESOLVED_STATIC_NAMES)

_ZARR_CLIMATE_ALIASES: dict[str, tuple[str, ...]] = {
    "tmax": ("tmax",),
    "tmin": ("tmin",),
    "tmean": ("tmean",),
    "precip": ("precip", "pr"),
    "srad": ("srad", "ssrd"),
    "vpd": ("vpd", "vpd_mean"),
    "et0": ("et0",),
    "sm_root": ("sm_root", "swvl1"),
    "wind10m": ("wind10m", "u10"),
    "rh_mean": ("rh_mean", "rh"),
    "co2_ppm": ("co2_ppm", "co2"),
}


@dataclass(frozen=True)
class ResolvedStaticFeatures:
    """Raw static covariates before mapping into the yield-surrogate 13-vector."""

    clay_pct: float
    sand_pct: float
    soc_gkg: float
    cec_cmolkg: float
    ph: float
    elevation_m: float
    slope_deg: float
    chirps_annual_mm: float
    protected_dist_km: float
    cocoa_prob: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.clay_pct,
                self.sand_pct,
                self.soc_gkg,
                self.cec_cmolkg,
                self.ph,
                self.elevation_m,
                self.slope_deg,
                self.chirps_annual_mm,
                self.protected_dist_km,
                self.cocoa_prob,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FeatureResolverConfig:
    era5_zarr_path: Path = DEFAULT_ERA5_ZARR
    static_zarr_path: Path = DEFAULT_STATIC_ZARR
    features_cache_zarr_path: Path = DEFAULT_FEATURES_CACHE_ZARR
    farm_registry_path: Path = DEFAULT_FARM_REGISTRY
    cache_dir: Path = DEFAULT_CACHE_DIR
    feature_store_root: Path | None = None
    use_real_features: bool = True
    use_galileo_embedding: bool = False
    galileo_embedding_dim: int = 128
    galileo_h3_resolution: int = 7
    gee_project: str | None = None
    cocoa_exposure_year: int = 2023
    cocoa_exposure_threshold: float = DEFAULT_THRESHOLD
    cocoa_exposure_backend: ExposureBackend = "ensemble_v2"
    ensemble_weights_path: Path = DEFAULT_ENSEMBLE_WEIGHTS_PATH
    agrifm_checkpoint_path: Path = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"
    grid_step_deg: float = GRID_ROUND_STEP


def _env_use_real_features() -> bool:
    raw = os.environ.get("USE_REAL_FEATURES", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def round_to_grid(lat: float, lon: float, step: float = GRID_ROUND_STEP) -> tuple[float, float]:
    """Round coordinates to a regular grid (default 0.05°)."""
    return (round(lat / step) * step, round(lon / step) * step)


def _lat_lon_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.coords:
        return "lat", "lon"
    raise ValueError(f"No lat/lon coordinates in dataset: {list(ds.coords)}")


def _pick_var(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {names} in dataset vars {list(ds.data_vars)}")


def _awc_mm_from_texture(sand_pct: float, clay_pct: float, depth_cm: float = 100.0) -> float:
    sand = np.clip(sand_pct, 1.0, 99.0) / 100.0
    clay = np.clip(clay_pct, 1.0, 99.0) / 100.0
    theta_s = 0.332 - 0.0007251 * sand * 100 + 0.1276 * math.log(clay * 100 + 1e-6)
    theta_fc = 0.2576 - 0.002 * sand * 100 - 0.00136 * clay * 100 + 0.2322 * theta_s
    awc_frac = max(0.05, theta_fc - 0.12)
    return float(np.clip(awc_frac * depth_cm * 10.0, 40.0, 280.0))


def _lookup_farm_registry(
    lat: float,
    lon: float,
    registry_path: Path,
) -> tuple[float, float]:
    if not registry_path.is_file():
        return DEFAULT_TREE_AGE_YEARS, DEFAULT_PLANTING_DENSITY

    import pandas as pd

    df = pd.read_parquet(registry_path)
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return DEFAULT_TREE_AGE_YEARS, DEFAULT_PLANTING_DENSITY

    dist2 = (df["lat"].astype(float) - lat) ** 2 + (df["lon"].astype(float) - lon) ** 2
    idx = int(dist2.idxmin())
    if float(dist2.loc[idx]) > FARM_REGISTRY_MAX_DIST_DEG2:
        return DEFAULT_TREE_AGE_YEARS, DEFAULT_PLANTING_DENSITY

    row = df.loc[idx]
    return (
        float(row.get("tree_age_years", DEFAULT_TREE_AGE_YEARS)),
        float(row.get("planting_density_trees_ha", DEFAULT_PLANTING_DENSITY)),
    )


def _cocoa_belt_probability(lat: float, lon: float) -> float:
    in_africa = -12.0 <= lat <= 12.0 and -12.0 <= lon <= 5.0
    in_americas = -15.0 <= lat <= 15.0 and -85.0 <= lon <= -30.0
    in_se_asia = -11.0 <= lat <= 7.0 and 95.0 <= lon <= 141.0
    if in_africa or in_americas or in_se_asia:
        return 0.75
    if abs(lat) <= 20.0:
        return 0.35
    return 0.05


def _climate_tensor_from_dataset(ds: xr.Dataset, year: int) -> np.ndarray:
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("Climate dataset missing time dimension")

    annual = ds.sel(time=ds["time"].dt.year == year)
    if int(annual.sizes.get("time", 0)) < SEQUENCE_LENGTH:
        annual = ds.sortby("time").isel(time=slice(-SEQUENCE_LENGTH, None))
    else:
        annual = annual.isel(time=slice(0, SEQUENCE_LENGTH))

    channels: list[np.ndarray] = []
    for name in CLIMATE_CHANNEL_NAMES:
        if name == "co2_ppm":
            if "co2_ppm" in annual.data_vars:
                values = np.asarray(annual["co2_ppm"].values, dtype=np.float32).reshape(-1)
            else:
                values = np.full(SEQUENCE_LENGTH, DEFAULT_CO2_PPM + (year - 2020) * 2.5, dtype=np.float32)
        else:
            var = _pick_var(annual, _ZARR_CLIMATE_ALIASES[name])
            values = np.asarray(annual[var].values, dtype=np.float32).reshape(-1)
        if values.size < SEQUENCE_LENGTH:
            pad = np.full(SEQUENCE_LENGTH - values.size, values[-1] if values.size else 0.0)
            values = np.concatenate([pad, values])
        elif values.size > SEQUENCE_LENGTH:
            values = values[-SEQUENCE_LENGTH:]
        channels.append(values)

    return np.stack(channels, axis=-1).astype(np.float32)


def climate_tensor_from_dataset_point(ds: xr.Dataset, lat: float, lon: float, year: int) -> Tensor:
    lat_name, lon_name = _lat_lon_coord_names(ds)
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    arr = _climate_tensor_from_dataset(point, year)
    return torch.from_numpy(arr).unsqueeze(0)


class _LRUArrayCache:
    """Simple LRU for numpy climate/static arrays keyed by rounded (lat, lon, year?)."""

    def __init__(self, maxsize: int = MEMORY_CACHE_MAX) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


class FarmFeatureResolver:
    """Climate, static site, and optional Galileo features for a farm location."""

    def __init__(self, config: FeatureResolverConfig | None = None) -> None:
        self.config = config or FeatureResolverConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_cache: Any = None
        try:
            from diskcache import Cache

            self._disk_cache = Cache(str(self.config.cache_dir))
        except Exception as exc:
            logger.warning("diskcache unavailable (%s); using in-memory LRU only", exc)
        self._memory_cache = _LRUArrayCache()
        self._galileo_extractor: Any | None = None
        self._feature_store: FeatureStore | None = None
        if self.config.feature_store_root is not None:
            try:
                self._feature_store = FeatureStore(self.config.feature_store_root)
            except Exception as exc:
                logger.warning("Failed to init FeatureStore (%s)", exc)

    @property
    def use_real_features(self) -> bool:
        return self.config.use_real_features

    def _grid_key(self, lat: float, lon: float, year: int | None = None) -> tuple[Any, ...]:
        lat_r, lon_r = round_to_grid(lat, lon, self.config.grid_step_deg)
        if year is None:
            return (lat_r, lon_r)
        return (lat_r, lon_r, year)

    def _cache_get_climate(self, key: tuple[Any, ...]) -> Tensor | None:
        arr = self._memory_cache.get(key)
        if arr is not None:
            return torch.from_numpy(arr).unsqueeze(0)
        if self._disk_cache is not None:
            cached = self._disk_cache.get(f"climate:{key}")
            if cached is not None:
                arr = np.asarray(cached, dtype=np.float32)
                self._memory_cache.set(key, arr)
                return torch.from_numpy(arr).unsqueeze(0)
        return None

    def _cache_set_climate(self, key: tuple[Any, ...], tensor: Tensor) -> None:
        arr = tensor.squeeze(0).numpy().astype(np.float32)
        self._memory_cache.set(key, arr)
        if self._disk_cache is not None:
            self._disk_cache.set(f"climate:{key}", arr)

    def _cache_get_static(self, key: tuple[float, float]) -> Tensor | None:
        arr = self._memory_cache.get(key)
        if arr is not None:
            return torch.from_numpy(arr).unsqueeze(0)
        if self._disk_cache is not None:
            cached = self._disk_cache.get(f"static:{key}")
            if cached is not None:
                arr = np.asarray(cached, dtype=np.float32)
                self._memory_cache.set(key, arr)
                return torch.from_numpy(arr).unsqueeze(0)
        return None

    def _cache_set_static(self, key: tuple[float, float], vec: np.ndarray) -> None:
        self._memory_cache.set(key, vec)
        if self._disk_cache is not None:
            self._disk_cache.set(f"static:{key}", vec)

    def resolve_climate(self, lat: float, lon: float, year: int) -> Tensor:
        """Daily climate ``[1, 365, 11]`` from cache / Zarr / GEE / geo_mock."""
        key = self._grid_key(lat, lon, year)
        hit = self._cache_get_climate(key)
        if hit is not None:
            return hit

        if not self.use_real_features:
            tensor = self._climate_from_geo_mock(lat, lon)
            self._cache_set_climate(key, tensor)
            return tensor

        cache_path = self.config.features_cache_zarr_path
        if cache_path.is_dir():
            try:
                tensor = self._climate_from_features_cache(cache_path, lat, lon, year)
                self._cache_set_climate(key, tensor)
                return tensor
            except Exception as exc:
                logger.debug("features_cache climate miss: %s", exc)

        if self._feature_store is not None:
            try:
                tensor = self._feature_store.climate_tensor(lat=lat, lon=lon, year=year)
                if torch.isfinite(tensor).all():
                    self._cache_set_climate(key, tensor)
                    return tensor
            except Exception as exc:
                logger.info("FeatureStore climate miss (%s)", exc)

        zarr_path = self.config.era5_zarr_path
        if zarr_path.is_dir():
            try:
                tensor = self._climate_from_zarr(zarr_path, lat, lon, year)
                self._cache_set_climate(key, tensor)
                return tensor
            except Exception as exc:
                logger.warning("ERA5 Zarr read failed (%s); trying GEE", exc)

        tensor = self._climate_from_gee(lat, lon, year)
        self._cache_set_climate(key, tensor)
        return tensor

    def resolve_static(self, lat: float, lon: float, year: int | None = None) -> Tensor:
        """Site static ``[1, 13]`` for the yield surrogate."""
        key = self._grid_key(lat, lon)
        hit = self._cache_get_static(key)
        if hit is not None:
            return hit

        if not self.use_real_features:
            vec = self._static_from_geo_mock(lat, lon)
            self._cache_set_static(key, vec)
            return torch.from_numpy(vec).unsqueeze(0)

        yr = year if year is not None else 2023
        cache_path = self.config.features_cache_zarr_path
        if cache_path.is_dir():
            try:
                vec = self._static_from_features_cache(cache_path, lat, lon)
                self._cache_set_static(key, vec)
                return torch.from_numpy(vec).unsqueeze(0)
            except Exception as exc:
                logger.debug("features_cache static miss: %s", exc)

        if self._feature_store is not None:
            try:
                raw = self._feature_store.static_vector(lat=lat, lon=lon).numpy().reshape(-1)
                if np.isfinite(raw).all() and raw.size == SITE_STATIC_DIM:
                    self._cache_set_static(key, raw.astype(np.float32))
                    return torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)
            except Exception as exc:
                logger.info("FeatureStore static miss (%s)", exc)

        zarr_path = self.config.static_zarr_path
        if zarr_path.is_dir():
            try:
                vec = self._static_from_legacy_static_zarr(zarr_path, lat, lon)
                self._cache_set_static(key, vec)
                return torch.from_numpy(vec).unsqueeze(0)
            except Exception as exc:
                logger.warning("site_static Zarr failed (%s); trying GEE", exc)

        resolved = self._resolve_static_from_gee(lat, lon, year=yr)
        vec = self._pack_model_static_vector(resolved, lat, lon)
        self._cache_set_static(key, vec)
        return torch.from_numpy(vec).unsqueeze(0)

    def resolve_static_with_galileo(self, lat: float, lon: float, year: int) -> Tensor:
        site = self.resolve_static(lat, lon, year=year)
        if not self.config.use_galileo_embedding:
            return site
        gal = self._resolve_galileo_embedding(lat, lon, year)
        return torch.cat([site, gal], dim=-1)

    def _climate_from_geo_mock(self, lat: float, lon: float) -> Tensor:
        from api.geo_mock import fetch_climate_and_soil

        climate, _ = fetch_climate_and_soil(lat, lon)
        return climate

    def _static_from_geo_mock(self, lat: float, lon: float) -> np.ndarray:
        from api.geo_mock import fetch_climate_and_soil

        _, static = fetch_climate_and_soil(lat, lon)
        return np.asarray(static.squeeze(0).numpy(), dtype=np.float32)

    def _climate_from_features_cache(self, path: Path, lat: float, lon: float, year: int) -> Tensor:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        lat_r, lon_r = round_to_grid(lat, lon, self.config.grid_step_deg)
        if "climate" not in ds.data_vars:
            raise KeyError("features_cache missing 'climate' variable")
        if "year" in ds["climate"].dims:
            block = ds["climate"].sel(year=year, **{lat_name: lat_r, lon_name: lon_r}, method="nearest")
        else:
            block = ds["climate"].sel(**{lat_name: lat_r, lon_name: lon_r}, method="nearest")
        arr = np.asarray(block.values, dtype=np.float32)
        if arr.shape != (SEQUENCE_LENGTH, len(CLIMATE_CHANNEL_NAMES)):
            raise ValueError(f"Unexpected climate shape {arr.shape}")
        return torch.from_numpy(arr).unsqueeze(0)

    def _static_from_features_cache(self, path: Path, lat: float, lon: float) -> np.ndarray:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        lat_r, lon_r = round_to_grid(lat, lon, self.config.grid_step_deg)
        point = ds.sel({lat_name: lat_r, lon_name: lon_r}, method="nearest")
        resolved = ResolvedStaticFeatures(
            clay_pct=float(point["clay_pct"]),
            sand_pct=float(point["sand_pct"]),
            soc_gkg=float(point["soc_gkg"]),
            cec_cmolkg=float(point["cec_cmolkg"]),
            ph=float(point["ph"]),
            elevation_m=float(point["elevation_m"]),
            slope_deg=float(point["slope_deg"]),
            chirps_annual_mm=float(point["chirps_annual_mm"]),
            protected_dist_km=float(point["protected_dist_km"]),
            cocoa_prob=float(point["cocoa_prob"]),
        )
        return self._pack_model_static_vector(resolved, lat, lon)

    def _climate_from_zarr(self, path: Path, lat: float, lon: float, year: int) -> Tensor:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        arr = _climate_tensor_from_dataset(point, year)
        return torch.from_numpy(arr).unsqueeze(0)

    def _climate_from_gee(self, lat: float, lon: float, year: int) -> Tensor:
        initialize_earth_engine(project=self.config.gee_project)
        point = ee.Geometry.Point([lon, lat])
        ingest = ERA5Ingest(
            point.buffer(2_000),
            f"{year}-01-01",
            f"{year}-12-31",
            project=self.config.gee_project,
        )
        ds = ingest.build()
        lat_name, lon_name = _lat_lon_coord_names(ds)
        point_ds = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        if "co2_ppm" not in point_ds.data_vars:
            point_ds["co2_ppm"] = xr.DataArray(
                np.full(int(point_ds.sizes["time"]), DEFAULT_CO2_PPM + (year - 2020) * 2.5),
                dims=["time"],
            )
        for name, default in (("wind10m", 2.0), ("rh_mean", 75.0)):
            if name not in point_ds.data_vars:
                point_ds[name] = xr.DataArray(
                    np.full(int(point_ds.sizes["time"]), default),
                    dims=["time"],
                )
        arr = _climate_tensor_from_dataset(point_ds, year)
        return torch.from_numpy(arr).unsqueeze(0)

    def _static_from_legacy_static_zarr(self, path: Path, lat: float, lon: float) -> np.ndarray:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        tree_age, density = _lookup_farm_registry(lat, lon, self.config.farm_registry_path)
        resolved = ResolvedStaticFeatures(
            clay_pct=float(point.get("clay_pct", 25.0)),
            sand_pct=float(point.get("sand_pct", 40.0)),
            soc_gkg=float(point.get("soc_gkg", 20.0)),
            cec_cmolkg=float(point.get("cec_cmolkg", 15.0)),
            ph=float(point.get("ph", 5.5)),
            elevation_m=float(point.get("elevation_m", 200.0)),
            slope_deg=float(point.get("slope_deg", 2.0)),
            chirps_annual_mm=float(point.get("chirps_annual_mm", 1200.0)),
            protected_dist_km=float(point.get("protected_dist_km", 50.0)),
            cocoa_prob=float(point.get("cocoa_prob", 0.5)),
        )
        return self._pack_model_static_vector(resolved, lat, lon, tree_age, density)

    def _resolve_static_from_gee(self, lat: float, lon: float, *, year: int) -> ResolvedStaticFeatures:
        initialize_earth_engine(project=self.config.gee_project)
        point = ee.Geometry.Point([lon, lat])
        fdp_year = (
            self.config.cocoa_exposure_year
            if self.config.cocoa_exposure_year in (2020, 2023)
            else (year if year in (2020, 2023) else 2023)
        )

        soil = ee.Image(SOILGRIDS_IMAGE)
        sand = soil.select("sand_0-5cm_mean")
        clay = soil.select("clay_0-5cm_mean")
        soc = soil.select("soc_0-5cm_mean")
        cec = soil.select("cec_0-5cm_mean")
        ph = soil.select("phh2o_0-5cm_mean")

        srtm = ee.Image(SRTM_IMAGE).select("elevation")
        slope = ee.Terrain.slope(srtm)

        chirps_mean = (
            ee.ImageCollection(CHIRPS_DAILY)
            .filterDate("2000-01-01", "2020-12-31")
            .select("precipitation")
            .mean()
            .rename("chirps_annual")
        )

        wdpa = ee.FeatureCollection(WDPA_IMAGE)
        protected_dist_m = point.distance(wdpa, maxError=500)

        stack = ee.Image.cat([sand, clay, soc, cec, ph, srtm, slope, chirps_mean]).rename(
            [
                "sand",
                "clay",
                "soc",
                "cec",
                "ph",
                "elev",
                "slope",
                "chirps_annual",
            ]
        )
        sample = stack.reduceRegion(ee.Reducer.first(), point, scale=250).getInfo() or {}
        dist_info = protected_dist_m.getInfo()
        protected_km = float(dist_info) / 1000.0 if dist_info is not None else 50.0

        backend = self.config.cocoa_exposure_backend
        cocoa_prob = sample_cocoa_probability_at_point(
            lat,
            lon,
            year=fdp_year,
            threshold=self.config.cocoa_exposure_threshold,
            backend=backend,
            project=self.config.gee_project,
            agrifm_checkpoint=self.config.agrifm_checkpoint_path,
            ensemble_weights_path=self.config.ensemble_weights_path,
        )

        return ResolvedStaticFeatures(
            clay_pct=float(sample.get("clay", 25.0)),
            sand_pct=float(sample.get("sand", 40.0)),
            soc_gkg=float(sample.get("soc", 20.0)) / 10.0,
            cec_cmolkg=float(sample.get("cec", 15.0)) / 10.0,
            ph=float(sample.get("ph", 55.0)) / 10.0,
            elevation_m=float(sample.get("elev", 200.0)),
            slope_deg=float(sample.get("slope", 2.0)),
            chirps_annual_mm=float(sample.get("chirps_annual", 1200.0)) * 365.0,
            protected_dist_km=float(np.clip(protected_km, 0.0, 500.0)),
            cocoa_prob=float(np.clip(cocoa_prob, 0.0, 1.0)),
        )

    def _pack_model_static_vector(
        self,
        resolved: ResolvedStaticFeatures,
        lat: float,
        lon: float,
        tree_age_years: float | None = None,
        planting_density_trees_ha: float | None = None,
    ) -> np.ndarray:
        """
        Map resolved static (SoilGrids / terrain / CHIRPS / WDPA / cocoa) into the
        13-d yield-surrogate layout (AWC at 0; simulation encodings at 2–4).
        """
        if tree_age_years is None or planting_density_trees_ha is None:
            tree_age_years, planting_density_trees_ha = _lookup_farm_registry(
                lat, lon, self.config.farm_registry_path
            )

        vec = np.zeros(SITE_STATIC_DIM, dtype=np.float32)
        vec[0] = _awc_mm_from_texture(resolved.sand_pct, resolved.clay_pct)
        vec[1] = np.clip(resolved.sand_pct / 100.0, 0.0, 1.0)
        vec[5] = np.clip(resolved.clay_pct / 100.0, 0.0, 1.0)
        vec[6] = np.clip(resolved.soc_gkg / 50.0, 0.0, 1.0)
        vec[7] = np.clip(resolved.ph / 14.0, 0.0, 1.0)
        vec[8] = np.clip(resolved.chirps_annual_mm / 3000.0, 0.0, 1.0)
        vec[9] = np.clip(resolved.cocoa_prob, 0.0, 1.0)
        age_norm, cohort, dens_norm = pack_tree_age_static(
            tree_age_years,
            planting_density_trees_ha=planting_density_trees_ha,
        )
        vec[10] = age_norm
        vec[11] = cohort
        vec[12] = dens_norm
        return vec

    def _resolve_galileo_embedding(self, lat: float, lon: float, year: int) -> Tensor:
        if not self.config.use_galileo_embedding:
            raise RuntimeError("Galileo embeddings disabled")

        import h3

        cell = h3.latlng_to_cell(lat, lon, self.config.galileo_h3_resolution)
        key = self._grid_key(lat, lon, year)
        arr = self._memory_cache.get(("galileo",) + key)
        if arr is not None:
            return torch.from_numpy(arr).unsqueeze(0)

        extractor = self._get_galileo_extractor()
        s2 = torch.zeros(1, 4, 8, 8, 10)
        emb = extractor.embed(s2=s2, months=torch.tensor([[6, 7]]))
        pooled = emb.mean(dim=0).numpy().astype(np.float32)
        dim = self.config.galileo_embedding_dim
        if pooled.size != dim:
            pooled = pooled[:dim] if pooled.size > dim else np.pad(pooled, (0, dim - pooled.size))
        self._memory_cache.set(("galileo",) + key, pooled)
        return torch.from_numpy(pooled).unsqueeze(0)

    def _get_galileo_extractor(self) -> Any:
        if self._galileo_extractor is None:
            from models.galileo_features import GalileoFeatureConfig, GalileoFeatureExtractor

            self._galileo_extractor = GalileoFeatureExtractor(
                GalileoFeatureConfig(device="cpu", size="nano"),
            )
        return self._galileo_extractor


def resolve_climate(
    lat: float,
    lon: float,
    year: int,
    *,
    resolver: FarmFeatureResolver | None = None,
    era5_zarr_path: Path | None = None,
) -> Tensor:
    """Module-level helper: daily climate ``[1, 365, 11]``."""
    res = resolver or FarmFeatureResolver(
        FeatureResolverConfig(era5_zarr_path=era5_zarr_path or DEFAULT_ERA5_ZARR)
    )
    return res.resolve_climate(lat, lon, year)


def resolve_static(
    lat: float,
    lon: float,
    year: int | None = None,
    *,
    resolver: FarmFeatureResolver | None = None,
) -> Tensor:
    """Module-level helper: static site vector ``[1, 13]``."""
    res = resolver or FarmFeatureResolver()
    return res.resolve_static(lat, lon, year=year)


def build_resolver_from_settings(settings: Any) -> FarmFeatureResolver:
    """Construct resolver from :class:`~api.config.APISettings`."""
    use_real = getattr(settings, "use_real_features", _env_use_real_features())
    return FarmFeatureResolver(
        FeatureResolverConfig(
            era5_zarr_path=Path(getattr(settings, "era5_zarr_path", DEFAULT_ERA5_ZARR)),
            static_zarr_path=Path(getattr(settings, "static_zarr_path", DEFAULT_STATIC_ZARR)),
            features_cache_zarr_path=Path(
                getattr(settings, "features_cache_zarr_path", DEFAULT_FEATURES_CACHE_ZARR)
            ),
            cache_dir=Path(getattr(settings, "feature_cache_dir", DEFAULT_CACHE_DIR)),
            feature_store_root=getattr(settings, "feature_store_root", None),
            use_real_features=bool(use_real),
            use_galileo_embedding=bool(getattr(settings, "use_galileo_embedding", False)),
            galileo_embedding_dim=int(getattr(settings, "galileo_embedding_dim", 128)),
            gee_project=getattr(settings, "earthengine_project", None),
            farm_registry_path=Path(
                getattr(settings, "farm_registry_path", DEFAULT_FARM_REGISTRY)
            ),
            cocoa_exposure_year=int(getattr(settings, "cocoa_exposure_year", 2023)),
            cocoa_exposure_threshold=float(
                getattr(settings, "cocoa_exposure_threshold", DEFAULT_THRESHOLD)
            ),
            cocoa_exposure_backend=_resolve_exposure_backend(settings),
            ensemble_weights_path=Path(
                getattr(settings, "ensemble_weights_path", DEFAULT_ENSEMBLE_WEIGHTS_PATH)
            ),
            agrifm_checkpoint_path=Path(
                getattr(settings, "agrifm_checkpoint_path", DEFAULT_AGRIFM_CHECKPOINT)
            ),
        )
    )


def _resolve_exposure_backend(settings: Any) -> ExposureBackend:
    """Map API settings to exposure backend (honours ENSEMBLE_BACKEND=v2)."""
    raw = getattr(settings, "cocoa_exposure_backend", "ensemble_v2")
    ensemble_mode = getattr(settings, "ensemble_backend", "v2")
    if raw in ("ensemble", "ensemble_v2") and ensemble_mode == "v2":
        return "ensemble_v2"
    return raw  # type: ignore[return-value]


__all__ = [
    "DEFAULT_ERA5_ZARR",
    "DEFAULT_FEATURES_CACHE_ZARR",
    "FarmFeatureResolver",
    "FeatureResolverConfig",
    "GRID_ROUND_STEP",
    "ISRIC_SOILGRIDS_DATA_URL",
    "RESOLVED_STATIC_NAMES",
    "ResolvedStaticFeatures",
    "SITE_STATIC_DIM",
    "build_resolver_from_settings",
    "climate_tensor_from_dataset_point",
    "resolve_climate",
    "resolve_static",
    "round_to_grid",
]
