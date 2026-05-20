"""
Resolve farm-level climate, static site, and optional Galileo features for API inference.

Climate: pre-exported Zarr at ``data/processed/era5_2020_2024.zarr`` with live
Earth Engine + Xee fallback (diskcache). Static: SoilGrids, SRTM, Hansen, Kalischek
cocoa suitability via GEE point samples (cached). Galileo: frozen embeddings keyed
by H3 cell when enabled.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ee
import numpy as np
import torch
import xarray as xr
import xee  # noqa: F401 — registers the ``ee`` Xarray backend
from diskcache import Cache
from torch import Tensor

from data.era5_ingest import ERA5Ingest
from data.gee_auth import initialize_earth_engine
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 365
SITE_STATIC_DIM = 10
DEFAULT_AWC_MM = 150.0
DEFAULT_CO2_PPM = 415.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA5_ZARR = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
DEFAULT_STATIC_ZARR = _REPO_ROOT / "data" / "processed" / "site_static.zarr"
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "cache" / "api_features"

# GEE asset IDs for static covariates
SOILGRIDS_IMAGE = "projects/soilgrids-isric/soilgrids_world"
HANSEN_IMAGE = "UMD/hansen/global_forest_change_2023_v1_11"
SRTM_IMAGE = "USGS/SRTMGL1_003"
# Kalischek et al. 2023 cocoa suitability — override via env if published to GEE
KALISCHEK_COCOA_ASSET: str | None = None

# Zarr variable aliases → surrogate channel name
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
class FeatureResolverConfig:
    era5_zarr_path: Path = DEFAULT_ERA5_ZARR
    static_zarr_path: Path = DEFAULT_STATIC_ZARR
    cache_dir: Path = DEFAULT_CACHE_DIR
    use_galileo_embedding: bool = False
    galileo_embedding_dim: int = 128
    galileo_h3_resolution: int = 7
    gee_project: str | None = None


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
    """Saxton–Rawls-style field-capacity water storage (mm) from texture (%)."""
    sand = np.clip(sand_pct, 1.0, 99.0) / 100.0
    clay = np.clip(clay_pct, 1.0, 99.0) / 100.0
    # Total porosity (fraction)
    theta_s = 0.332 - 0.0007251 * sand * 100 + 0.1276 * math.log(clay * 100 + 1e-6)
    theta_fc = 0.2576 - 0.002 * sand * 100 - 0.00136 * clay * 100 + 0.2322 * theta_s
    awc_frac = max(0.05, theta_fc - 0.12)
    return float(np.clip(awc_frac * depth_cm * 10.0, 40.0, 280.0))


def _cocoa_belt_probability(lat: float, lon: float) -> float:
    """Heuristic cocoa suitability when Kalischek raster is unavailable."""
    # West Africa belt + Americas tropics
    in_africa = -12.0 <= lat <= 12.0 and -12.0 <= lon <= 5.0
    in_americas = -15.0 <= lat <= 15.0 and -85.0 <= lon <= -30.0
    if in_africa or in_americas:
        return 0.75
    if abs(lat) <= 20.0:
        return 0.35
    return 0.05


def _climate_tensor_from_dataset(ds: xr.Dataset, year: int) -> np.ndarray:
    """Build ``[365, 11]`` float32 array in :data:`CLIMATE_CHANNEL_NAMES` order."""
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


class FarmFeatureResolver:
    """Climate, static site, and optional Galileo features for a farm location."""

    def __init__(self, config: FeatureResolverConfig | None = None) -> None:
        self.config = config or FeatureResolverConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.config.cache_dir))
        self._galileo_extractor: Any | None = None

    def resolve_climate(self, lat: float, lon: float, year: int) -> Tensor:
        """
        Daily climate stack ``[1, 365, 11]`` (see :data:`~models.yield_surrogate.CLIMATE_CHANNEL_NAMES`).
        """
        cache_key = f"climate:{lat:.6f}:{lon:.6f}:{year}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return torch.from_numpy(np.asarray(cached, dtype=np.float32)).unsqueeze(0)

        zarr_path = self.config.era5_zarr_path
        if zarr_path.is_dir():
            try:
                tensor = self._climate_from_zarr(zarr_path, lat, lon, year)
                self._cache.set(cache_key, tensor.numpy())
                return tensor
            except Exception as exc:
                logger.warning("ERA5 Zarr read failed (%s); falling back to live GEE", exc)

        tensor = self._climate_from_gee(lat, lon, year)
        self._cache.set(cache_key, tensor.numpy())
        return tensor

    def resolve_static(self, lat: float, lon: float) -> Tensor:
        """Site static covariates ``[1, 10]`` (AWC + soil/terrain/cocoa; indices 2–4 for API encoding)."""
        cache_key = f"static:{lat:.6f}:{lon:.6f}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return torch.from_numpy(np.asarray(cached, dtype=np.float32)).unsqueeze(0)

        zarr_path = self.config.static_zarr_path
        if zarr_path.is_dir():
            try:
                vec = self._static_from_zarr(zarr_path, lat, lon)
                self._cache.set(cache_key, vec)
                return torch.from_numpy(vec).unsqueeze(0)
            except Exception as exc:
                logger.warning("Static Zarr read failed (%s); falling back to GEE", exc)

        vec = self._static_from_gee(lat, lon)
        self._cache.set(cache_key, vec)
        return torch.from_numpy(vec).unsqueeze(0)

    def resolve_galileo_embedding(self, lat: float, lon: float, year: int) -> Tensor:
        """
        Pooled Galileo embedding ``[1, D]`` (cached by H3 cell).

        Requires ``use_galileo_embedding`` and a loaded :class:`~models.galileo_features.GalileoFeatureExtractor`.
        """
        if not self.config.use_galileo_embedding:
            raise RuntimeError("Galileo embeddings disabled (USE_GALILEO_EMBEDDING=false)")

        import h3

        cell = h3.latlng_to_cell(lat, lon, self.config.galileo_h3_resolution)
        cache_key = f"galileo:{cell}:{year}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return torch.from_numpy(np.asarray(cached, dtype=np.float32)).unsqueeze(0)

        extractor = self._get_galileo_extractor()
        # Minimal placeholder inputs — production path should wire real Sentinel/ERA5 stacks
        s2 = torch.zeros(1, 4, 8, 8, 10)
        emb = extractor.embed(s2=s2, months=torch.tensor([[6, 7]]))
        pooled = emb.mean(dim=0).numpy().astype(np.float32)
        dim = self.config.galileo_embedding_dim
        if pooled.size != dim:
            if pooled.size > dim:
                pooled = pooled[:dim]
            else:
                pooled = np.pad(pooled, (0, dim - pooled.size))

        self._cache.set(cache_key, pooled)
        return torch.from_numpy(pooled).unsqueeze(0)

    def resolve_static_with_galileo(
        self,
        lat: float,
        lon: float,
        year: int,
    ) -> Tensor:
        """Concatenate site static ``[1,10]`` with Galileo ``[1,D]`` when enabled."""
        site = self.resolve_static(lat, lon)
        if not self.config.use_galileo_embedding:
            return site
        gal = self.resolve_galileo_embedding(lat, lon, year)
        return torch.cat([site, gal], dim=-1)

    def _climate_from_zarr(self, path: Path, lat: float, lon: float, year: int) -> Tensor:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        arr = _climate_tensor_from_dataset(point, year)
        return torch.from_numpy(arr).unsqueeze(0)

    def _climate_from_gee(self, lat: float, lon: float, year: int) -> Tensor:
        initialize_earth_engine(project=self.config.gee_project)
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(2_000)
        ingest = ERA5Ingest(
            aoi,
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
        # Ensure wind/rh if missing from minimal ingest
        for name, default in (("wind10m", 2.0), ("rh_mean", 75.0)):
            if name not in point_ds.data_vars:
                point_ds[name] = xr.DataArray(
                    np.full(int(point_ds.sizes["time"]), default),
                    dims=["time"],
                )
        arr = _climate_tensor_from_dataset(point_ds, year)
        return torch.from_numpy(arr).unsqueeze(0)

    def _static_from_zarr(self, path: Path, lat: float, lon: float) -> np.ndarray:
        ds = xr.open_zarr(path, consolidated=True)
        lat_name, lon_name = _lat_lon_coord_names(ds)
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        return self._pack_static_vector(
            sand_pct=float(point.get("sand_pct", 40.0)),
            clay_pct=float(point.get("clay_pct", 25.0)),
            soc_gkg=float(point.get("soc_gkg", 20.0)),
            ph=float(point.get("ph", 5.5)),
            elevation_m=float(point.get("elevation_m", 200.0)),
            slope_deg=float(point.get("slope_deg", 2.0)),
            treecover_pct=float(point.get("treecover_pct", 60.0)),
            cocoa_prob=float(point.get("cocoa_prob", 0.5)),
        )

    def _static_from_gee(self, lat: float, lon: float) -> np.ndarray:
        initialize_earth_engine(project=self.config.gee_project)
        point = ee.Geometry.Point([lon, lat])

        soil = ee.Image(SOILGRIDS_IMAGE)
        sand = soil.select("sand_0-5cm_mean").rename("sand")
        clay = soil.select("clay_0-5cm_mean").rename("clay")
        soc = soil.select("soc_0-5cm_mean").rename("soc")
        ph = soil.select("phh2o_0-5cm_mean").rename("ph")

        srtm = ee.Image(SRTM_IMAGE).select("elevation")
        slope = ee.Terrain.slope(srtm)
        hansen = ee.Image(HANSEN_IMAGE).select("treecover2000")

        stack = ee.Image.cat([sand, clay, soc, ph, srtm, slope, hansen]).rename(
            ["sand", "clay", "soc", "ph", "elev", "slope", "treecover"]
        )

        if KALISCHEK_COCOA_ASSET:
            cocoa = ee.Image(KALISCHEK_COCOA_ASSET).rename("cocoa")
            stack = stack.addBands(cocoa)

        sample = stack.reduceRegion(ee.Reducer.first(), point, scale=250).getInfo() or {}

        sand_pct = float(sample.get("sand", 40.0))
        clay_pct = float(sample.get("clay", 25.0))
        soc_gkg = float(sample.get("soc", 20.0)) / 10.0  # SoilGrids dg/kg → g/kg approx
        ph_val = float(sample.get("ph", 55.0)) / 10.0
        elev = float(sample.get("elev", 200.0))
        slope_deg = float(sample.get("slope", 2.0))
        treecover = float(sample.get("treecover", 60.0))
        cocoa_prob = float(sample.get("cocoa", _cocoa_belt_probability(lat, lon)))

        return self._pack_static_vector(
            sand_pct=sand_pct,
            clay_pct=clay_pct,
            soc_gkg=soc_gkg,
            ph=ph_val,
            elevation_m=elev,
            slope_deg=slope_deg,
            treecover_pct=treecover,
            cocoa_prob=cocoa_prob,
        )

    def _pack_static_vector(
        self,
        *,
        sand_pct: float,
        clay_pct: float,
        soc_gkg: float,
        ph: float,
        elevation_m: float,
        slope_deg: float,
        treecover_pct: float,
        cocoa_prob: float,
    ) -> np.ndarray:
        """
        Pack site static features (10).

        Index 0: AWC (mm, from texture). Indices 2–4 reserved for simulation encodings.
        """
        del elevation_m, slope_deg  # inform AWC / future extensions
        vec = np.zeros(SITE_STATIC_DIM, dtype=np.float32)
        vec[0] = _awc_mm_from_texture(sand_pct, clay_pct)
        vec[1] = np.clip(sand_pct / 100.0, 0.0, 1.0)
        vec[5] = np.clip(clay_pct / 100.0, 0.0, 1.0)
        vec[6] = np.clip(soc_gkg / 50.0, 0.0, 1.0)
        vec[7] = np.clip(ph / 14.0, 0.0, 1.0)
        vec[8] = np.clip(treecover_pct / 100.0, 0.0, 1.0)
        vec[9] = np.clip(cocoa_prob, 0.0, 1.0)
        return vec

    def _get_galileo_extractor(self) -> Any:
        if self._galileo_extractor is None:
            from models.galileo_features import GalileoFeatureConfig, GalileoFeatureExtractor

            self._galileo_extractor = GalileoFeatureExtractor(
                GalileoFeatureConfig(device="cpu", size="nano"),
            )
        return self._galileo_extractor


def build_resolver_from_settings(settings: Any) -> FarmFeatureResolver:
    """Construct resolver from :class:`~api.config.APISettings`."""
    return FarmFeatureResolver(
        FeatureResolverConfig(
            era5_zarr_path=Path(getattr(settings, "era5_zarr_path", DEFAULT_ERA5_ZARR)),
            static_zarr_path=Path(getattr(settings, "static_zarr_path", DEFAULT_STATIC_ZARR)),
            cache_dir=Path(getattr(settings, "feature_cache_dir", DEFAULT_CACHE_DIR)),
            use_galileo_embedding=bool(getattr(settings, "use_galileo_embedding", False)),
            galileo_embedding_dim=int(getattr(settings, "galileo_embedding_dim", 128)),
            gee_project=getattr(settings, "earthengine_project", None),
        )
    )
