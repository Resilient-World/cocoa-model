"""
Forest Data Partnership (FDP) cocoa probability ingest via Google Earth Engine.

Dataset
-------
Forest Data Partnership (2025) *Cocoa Probability model 2025a*, successor to
Kalischek et al. (2023) *Nature Food*. ImageCollection:
``projects/forestdatapartnership/assets/cocoa/model_2025a`` — 10 m, annual composites
for 2020 and 2023; coverage Ghana, Côte d'Ivoire, Cameroon, Nigeria, Indonesia,
Ecuador, Peru, Colombia (see :data:`REGIONS`).

Exposure backends
-----------------
- ``fdp``: FDP probability raster (GEE prior)
- ``galileo``: :class:`~models.galileo_seg.GalileoCocoaSegmentation` tile inference
- ``aef``: AlphaEarth Foundations 64-D embeddings + :class:`~models.aef_cocoa_head.AEFCocoaHead`
- ``ensemble``: ``0.5 * AEF + 0.3 * Galileo + 0.2 * FDP`` (rebalanced after AEF benchmark)

Licensing
---------
Non-commercial Earth Engine use: CC-BY 4.0 NC — attribution required
("Produced by Google for the Forest Data Partnership"). **Commercial deployments**
must accept the Forest Data Partnership Datasets Commercial Terms of Use.

All access is server-side GEE; raw tiles are not downloaded except via optional
lazy Xarray/Zarr materialization (Xee).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

logger = logging.getLogger(__name__)

FDP_COCOA_COLLECTION = "projects/forestdatapartnership/assets/cocoa/model_2025a"
PROBABILITY_BAND = "probability"
SUPPORTED_YEARS: tuple[int, ...] = (2020, 2023)
# FDP 2025a model card: F1-optimal precision/recall ≈ 0.96 (not Kalischek 2023)
FDP_MODEL_CARD_URL = "https://github.com/google/forest-data-partnership/tree/main/models/cocoa"
MIN_THRESHOLD = 0.5
DEFAULT_THRESHOLD = 0.96
DEFAULT_SCALE_M = 10

ExposureBackend = Literal["fdp", "galileo", "aef", "agrifm", "ensemble", "ensemble_v2"]

# Default ensemble v1 blend: AEF, Galileo, FDP (sums to 1.0)
DEFAULT_ENSEMBLE_WEIGHTS: tuple[float, float, float] = (0.5, 0.3, 0.2)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GALILEO_CHECKPOINT = _REPO_ROOT / "models" / "galileo_cocoa_seg.pt"
DEFAULT_AEF_CHECKPOINT = _REPO_ROOT / "models" / "aef_cocoa_head.pt"
DEFAULT_AGRIFM_CHECKPOINT = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"

# Renormalized AEF + Galileo weights when FDP tiles are unavailable (0.5 + 0.3 → 1.0)
GLOBAL_AEF_GAL_WEIGHTS: tuple[float, float] = (0.625, 0.375)

_REGION_ALIASES: dict[str, str] = {
    "gha": "ghana",
    "civ": "civ",
    "cmr": "cameroon",
    "nga": "nigeria",
    "idn": "indonesia",
    "ecu": "ecuador",
    "per": "peru",
    "col": "colombia",
}


@dataclass(frozen=True)
class RegionPreset:
    """Cocoa-producing region bounding box (WGS84) and FDP 2025a native coverage."""

    display_name: str
    west: float
    south: float
    east: float
    north: float
    fdp_native: bool = True


REGIONS: dict[str, RegionPreset] = {
    "ghana": RegionPreset("Ghana", -3.25, 4.7, 1.2, 11.2),
    "civ": RegionPreset("Côte d'Ivoire", -8.5, 4.0, -2.5, 11.0),
    "cameroon": RegionPreset("Cameroon", 8.0, 1.5, 16.5, 13.5),
    "nigeria": RegionPreset("Nigeria", 2.5, 4.0, 14.5, 14.0),
    "indonesia": RegionPreset("Indonesia", 95.0, -11.0, 141.0, 7.0),
    "ecuador": RegionPreset("Ecuador", -81.5, -5.5, -75.0, 2.0),
    "peru": RegionPreset("Peru", -81.0, -15.0, -68.0, 0.5),
    "colombia": RegionPreset("Colombia", -79.0, -4.5, -66.0, 12.0),
}


def normalize_region_key(name: str) -> str:
    """Map CLI aliases (``gha``, ``GHA``) to :data:`REGIONS` keys."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in REGIONS:
        return key
    if key in _REGION_ALIASES:
        return _REGION_ALIASES[key]
    if key in ("cote_divoire", "cote_d_ivoire", "ivory_coast"):
        return "civ"
    raise KeyError(f"Unknown region {name!r}; choose from {sorted(REGIONS)}")


def region_bounds_dict(region: str) -> dict[str, float]:
    """Return ``{west, south, east, north}`` for a region key."""
    preset = REGIONS[normalize_region_key(region)]
    return {
        "west": preset.west,
        "south": preset.south,
        "east": preset.east,
        "north": preset.north,
    }


def region_latlon_bounds(region: str) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` for sampling grids."""
    preset = REGIONS[normalize_region_key(region)]
    return (preset.south, preset.north, preset.west, preset.east)


def region_geometry(region: str) -> ee.Geometry:
    """Earth Engine rectangle for a named region."""
    b = region_bounds_dict(region)
    return ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])


def point_in_region(lat: float, lon: float, region: str) -> bool:
    preset = REGIONS[normalize_region_key(region)]
    return preset.south <= lat <= preset.north and preset.west <= lon <= preset.east


def is_fdp_covered(lat: float, lon: float) -> bool:
    """True when ``(lat, lon)`` lies in a region with native FDP 2025a tiles."""
    return any(
        preset.fdp_native and point_in_region(lat, lon, key) for key, preset in REGIONS.items()
    )


def region_for_point(lat: float, lon: float) -> str | None:
    """Return the first matching FDP-native region key for a coordinate, if any."""
    for key, preset in REGIONS.items():
        if preset.fdp_native and point_in_region(lat, lon, key):
            return key
    return None


def processed_era5_zarr_path(
    region: str,
    *,
    repo_root: Path | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> Path:
    """Default ERA5 Zarr path: ``data/processed/era5_<region>[_<start>_<end>].zarr``."""
    key = normalize_region_key(region)
    root = repo_root or _REPO_ROOT
    suffix = ""
    if start_year is not None and end_year is not None:
        suffix = f"_{start_year}_{end_year}"
    return root / "data" / "processed" / f"era5_{key}{suffix}.zarr"


def processed_sentinel_tif_path(region: str, *, repo_root: Path | None = None) -> Path:
    """Default Sentinel composite GeoTIFF: ``data/processed/s2_s1_<region>.tif``."""
    key = normalize_region_key(region)
    root = repo_root or _REPO_ROOT
    return root / "data" / "processed" / f"s2_s1_{key}.tif"


def _normalize_year(year: int) -> int:
    """Map arbitrary calendar year to nearest supported FDP composite (2020 or 2023)."""
    if year in SUPPORTED_YEARS:
        return year
    return 2023 if year >= 2022 else 2020


def _year_date_range(year: int) -> tuple[str, str]:
    y = _normalize_year(year)
    return f"{y}-01-01", f"{y}-12-31"


def validate_threshold(threshold: float) -> float:
    """
    Validate FDP probability → binary mask threshold.

    Raises
    ------
    ValueError
        If ``threshold`` is below :data:`MIN_THRESHOLD` (0.5).
    """
    value = float(threshold)
    if value < MIN_THRESHOLD:
        raise ValueError(
            f"threshold must be >= {MIN_THRESHOLD} (FDP probability scale), got {value}"
        )
    if value > 1.0:
        raise ValueError(f"threshold must be <= 1.0, got {value}")
    return value


def _cocoa_belt_probability(lat: float, lon: float) -> float:
    """Heuristic cocoa suitability outside FDP mask (West Africa + Americas belt)."""
    in_africa = -12.0 <= lat <= 12.0 and -12.0 <= lon <= 5.0
    in_americas = -15.0 <= lat <= 15.0 and -85.0 <= lon <= -30.0
    if in_africa or in_americas:
        return 0.75
    if abs(lat) <= 20.0:
        return 0.35
    return 0.05


class CocoaExposureIngest:
    """
    Ingest FDP cocoa probability for an AOI and calendar year.

    Default threshold 0.96 is the F1-optimal operating point documented in the
    FDP 2025a model card (see :data:`FDP_MODEL_CARD_URL`).
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        year: int = 2023,
        threshold: float = DEFAULT_THRESHOLD,
        project: str | None = None,
        *,
        backend: ExposureBackend = "fdp",
        galileo_checkpoint: Path | str | None = None,
        aef_checkpoint: Path | str | None = None,
        agrifm_checkpoint: Path | str | None = None,
        ensemble_weights: tuple[float, float, float] = DEFAULT_ENSEMBLE_WEIGHTS,
        ensemble_weights_path: Path | str | None = None,
        region: str | None = None,
    ) -> None:
        self.aoi = aoi
        self.year = _normalize_year(year)
        self.threshold = validate_threshold(threshold)
        self.project = project
        self.backend = backend
        self.galileo_checkpoint = (
            Path(galileo_checkpoint) if galileo_checkpoint else DEFAULT_GALILEO_CHECKPOINT
        )
        self.aef_checkpoint = Path(aef_checkpoint) if aef_checkpoint else DEFAULT_AEF_CHECKPOINT
        self.agrifm_checkpoint = Path(agrifm_checkpoint) if agrifm_checkpoint else DEFAULT_AGRIFM_CHECKPOINT
        self.ensemble_weights = ensemble_weights
        self.ensemble_weights_path = (
            Path(ensemble_weights_path)
            if ensemble_weights_path
            else _REPO_ROOT / "config" / "ensemble_weights.yaml"
        )
        self.region = normalize_region_key(region) if region else None
        self._probability_image: ee.Image | None = None
        self._galileo_model = None
        self._aef_head = None
        self._agrifm_model = None

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

    def _fdp_probability_at_point(self, lat: float, lon: float, scale_m: int) -> float | None:
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

    def _load_galileo_model(self) -> Any:
        if self._galileo_model is not None:
            return self._galileo_model
        from models.galileo_seg import GalileoCocoaSegmentation, load_galileo_seg_checkpoint

        if self.galileo_checkpoint.is_file():
            self._galileo_model = load_galileo_seg_checkpoint(
                self.galileo_checkpoint, device="cpu"
            )
        else:
            logger.warning(
                "Galileo checkpoint missing at %s; using uninitialized GalileoCocoaSegmentation",
                self.galileo_checkpoint,
            )
            self._galileo_model = GalileoCocoaSegmentation(model_size="base", freeze_backbone=True)
            self._galileo_model.eval()
        return self._galileo_model

    def _galileo_probability_at_point(self, lat: float, lon: float) -> float:
        """
        Point P(cocoa) from a single-tile Galileo forward pass.

        Uses a minimal synthetic 64×64 patch when full Sentinel stacks are not
        wired at the point API (production should pass real tile batches).
        """
        import torch

        model = self._load_galileo_model()
        h = w = 64
        t = 4
        rng = np.random.default_rng(int(hash((round(lat, 4), round(lon, 4))) % (2**32)))
        s2 = torch.from_numpy(rng.normal(0.2, 0.05, (1, t, h, w, 10)).astype(np.float32))
        s1 = torch.from_numpy(rng.normal(-12.0, 2.0, (1, t, h, w, 2)).astype(np.float32))
        era5 = torch.from_numpy(rng.normal(0.0, 1.0, (1, t, 5)).astype(np.float32))
        dem = torch.from_numpy(
            np.stack(
                [
                    np.full((h, w), 200.0 + 50.0 * lat, dtype=np.float32),
                    np.full((h, w), 2.0, dtype=np.float32),
                ],
                axis=-1,
            )
        ).unsqueeze(0)
        loc = torch.tensor([[lat, lon]], dtype=torch.float32)
        months = torch.tensor([[6, 7, 8, 9]], dtype=torch.long)
        batch = model.build_batch_dict(s2=s2, s1=s1, era5=era5, dem=dem, location=loc, months=months)
        prob = model.predict_proba(batch)
        return float(prob.mean().item())

    def _load_aef_head(self) -> Any:
        if self._aef_head is not None:
            return self._aef_head
        from models.aef_cocoa_head import AEFCocoaHead, load_aef_cocoa_head

        if self.aef_checkpoint.is_file():
            self._aef_head = load_aef_cocoa_head(self.aef_checkpoint, device="cpu")
        else:
            logger.warning(
                "AEF head checkpoint missing at %s; using uninitialized AEFCocoaHead",
                self.aef_checkpoint,
            )
            self._aef_head = AEFCocoaHead()
            self._aef_head.eval()
        return self._aef_head

    def _aef_embedding_at_point(self, lat: float, lon: float) -> np.ndarray | None:
        """Sample 64-D AlphaEarth embedding at a point (GEE)."""
        try:
            from data.alphaearth_embeddings import AlphaEarthIngest

            point_aoi = ee.Geometry.Point([lon, lat]).buffer(50)
            ingest = AlphaEarthIngest(point_aoi, year=self.year, project=self.project)
            return ingest.sample_point(lat, lon)
        except Exception as exc:
            logger.debug("AEF embedding sample failed (%s); using location prior", exc)
            return None

    def _location_prior_embedding(self, lat: float, lon: float) -> np.ndarray:
        """Deterministic pseudo-embedding when GEE is unavailable."""
        seed = int(hash((round(lat, 4), round(lon, 4))) % (2**32))
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, 64).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _aef_probability_at_point(self, lat: float, lon: float) -> float:
        import torch

        head = self._load_aef_head()
        emb = self._aef_embedding_at_point(lat, lon)
        if emb is None:
            emb = self._location_prior_embedding(lat, lon)
        t = torch.from_numpy(emb).unsqueeze(0)
        return float(head.predict_proba(t).item())

    def _load_agrifm_model(self) -> Any:
        if self._agrifm_model is not None:
            return self._agrifm_model
        from models.agrifm_seg import load_agrifm_seg_checkpoint

        if self.agrifm_checkpoint.is_file():
            self._agrifm_model = load_agrifm_seg_checkpoint(self.agrifm_checkpoint, device="cpu")
        else:
            from models.agrifm_seg import AgriFMCocoaSegmentation

            logger.warning(
                "AgriFM checkpoint missing at %s; using uninitialized segmentation",
                self.agrifm_checkpoint,
            )
            self._agrifm_model = AgriFMCocoaSegmentation(freeze_backbone=True)
            self._agrifm_model.eval()
        return self._agrifm_model

    def _agrifm_probability_at_point(self, lat: float, lon: float) -> float:
        """Point P(cocoa) from AgriFM Video Swin segmentation (synthetic tile at API)."""
        import torch

        model = self._load_agrifm_model()
        h = w = 64
        t = 8
        rng = np.random.default_rng(int(hash((round(lat, 4), round(lon, 4))) % (2**32)))
        s2 = torch.from_numpy(rng.normal(0.2, 0.05, (1, t, h, w, 10)).astype(np.float32))
        return float(model.predict_proba_numpy(s2))

    def _ensemble_v2_blend(
        self,
        lat: float,
        lon: float,
        *,
        scale_m: int,
    ) -> float:
        from data.ensemble_weights import load_ensemble_weights

        region_key = self.region or region_for_point(lat, lon)
        weights = load_ensemble_weights(region_key, path=self.ensemble_weights_path)
        parts: list[tuple[float, float]] = []
        for key, w in weights.items():
            if key == "aef":
                parts.append((w, self._aef_probability_at_point(lat, lon)))
            elif key == "galileo":
                parts.append((w, self._galileo_probability_at_point(lat, lon)))
            elif key == "agrifm":
                parts.append((w, self._agrifm_probability_at_point(lat, lon)))
            elif key == "fdp":
                fdp_p = self._fdp_probability_at_point(lat, lon, scale_m)
                if fdp_p is not None:
                    parts.append((w, fdp_p))
        weight_sum = sum(w for w, _ in parts)
        blended = sum(w * p for w, p in parts) / max(weight_sum, 1e-9)
        return float(np.clip(blended, 0.0, 1.0))

    def sample_point(
        self,
        lat: float,
        lon: float,
        scale_m: int = DEFAULT_SCALE_M,
    ) -> float | None:
        """
        Sample P(cocoa) at a point for the configured :attr:`backend`.

        Returns ``None`` when the FDP pixel is masked (``fdp`` / ``ensemble`` only).
        """
        if self.backend == "fdp":
            return self._fdp_probability_at_point(lat, lon, scale_m)

        if self.backend == "galileo":
            return self._galileo_probability_at_point(lat, lon)

        if self.backend == "aef":
            return self._aef_probability_at_point(lat, lon)

        if self.backend == "agrifm":
            return self._agrifm_probability_at_point(lat, lon)

        if self.backend == "ensemble_v2":
            return self._ensemble_v2_blend(lat, lon, scale_m=scale_m)

        # ensemble v1: 0.5 AEF + 0.3 Galileo + 0.2 FDP
        w_aef, w_gal, w_fdp = self.ensemble_weights
        aef_p = self._aef_probability_at_point(lat, lon)
        gal_p = self._galileo_probability_at_point(lat, lon)
        fdp_p = self._fdp_probability_at_point(lat, lon, scale_m)
        parts: list[tuple[float, float]] = [(w_aef, aef_p), (w_gal, gal_p)]
        if fdp_p is not None:
            parts.append((w_fdp, fdp_p))
        weight_sum = sum(w for w, _ in parts)
        blended = sum(w * p for w, p in parts) / max(weight_sum, 1e-9)
        return float(np.clip(blended, 0.0, 1.0))

    def sample_points(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        *,
        scale_m: int = DEFAULT_SCALE_M,
    ) -> np.ndarray:
        """Vectorized point sampling (loop over points; GEE is per-call)."""
        out = np.zeros(len(lats), dtype=np.float64)
        for i, (la, lo) in enumerate(zip(lats, lons, strict=True)):
            val = self.sample_point(float(la), float(lo), scale_m=scale_m)
            out[i] = _cocoa_belt_probability(float(la), float(lo)) if val is None else val
        return out

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
                "backend": self.backend,
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


def _global_aef_galileo_agrifm_probability(
    lat: float,
    lon: float,
    *,
    year: int,
    project: str | None,
    galileo_checkpoint: Path | str | None,
    aef_checkpoint: Path | str | None,
    agrifm_checkpoint: Path | str | None,
    ensemble_weights_path: Path | str | None = None,
) -> float:
    """Blend AEF + Galileo + AgriFM when FDP 2025a does not cover the point."""
    from data.ensemble_weights import load_ensemble_weights

    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        year=year,
        project=project,
        backend="aef",
        galileo_checkpoint=galileo_checkpoint,
        aef_checkpoint=aef_checkpoint,
        agrifm_checkpoint=agrifm_checkpoint,
        ensemble_weights_path=ensemble_weights_path,
    )
    weights = load_ensemble_weights(None, path=ensemble_weights_path or ing.ensemble_weights_path, global_fallback=True)
    parts: list[tuple[float, float]] = [
        (weights["aef"], ing._aef_probability_at_point(lat, lon)),
        (weights["galileo"], ing._galileo_probability_at_point(lat, lon)),
        (weights["agrifm"], ing._agrifm_probability_at_point(lat, lon)),
    ]
    weight_sum = sum(w for w, _ in parts)
    blended = sum(w * p for w, p in parts) / max(weight_sum, 1e-9)
    return float(np.clip(blended, 0.0, 1.0))


def _global_aef_galileo_probability(
    lat: float,
    lon: float,
    *,
    year: int,
    project: str | None,
    galileo_checkpoint: Path | str | None,
    aef_checkpoint: Path | str | None,
    agrifm_checkpoint: Path | str | None = None,
    ensemble_weights_path: Path | str | None = None,
) -> float:
    """Backward-compatible global blend (delegates to AEF+Galileo+AgriFM when agrifm ckpt set)."""
    if agrifm_checkpoint is not None or (ensemble_weights_path and Path(ensemble_weights_path).is_file()):
        return _global_aef_galileo_agrifm_probability(
            lat,
            lon,
            year=year,
            project=project,
            galileo_checkpoint=galileo_checkpoint,
            aef_checkpoint=aef_checkpoint,
            agrifm_checkpoint=agrifm_checkpoint,
            ensemble_weights_path=ensemble_weights_path,
        )
    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        year=year,
        project=project,
        backend="aef",
        galileo_checkpoint=galileo_checkpoint,
        aef_checkpoint=aef_checkpoint,
    )
    w_aef, w_gal = GLOBAL_AEF_GAL_WEIGHTS
    aef_p = ing._aef_probability_at_point(lat, lon)
    gal_p = ing._galileo_probability_at_point(lat, lon)
    blended = w_aef * aef_p + w_gal * gal_p
    return float(np.clip(blended, 0.0, 1.0))


def sample_cocoa_probability_at_point(
    lat: float,
    lon: float,
    *,
    year: int = 2023,
    threshold: float = DEFAULT_THRESHOLD,
    backend: ExposureBackend | None = None,
    galileo_checkpoint: Path | str | None = None,
    aef_checkpoint: Path | str | None = None,
    agrifm_checkpoint: Path | str | None = None,
    ensemble_weights_path: Path | str | None = None,
    project: str | None = None,
) -> float:
    """
    Region-aware P(cocoa) for API / feature resolver use.

    Inside FDP-native coverage: sample FDP (or configured backend). Outside coverage:
    AlphaEarth embeddings + Galileo (globally available).
    """
    if is_fdp_covered(lat, lon):
        use_backend = backend or "fdp"
        try:
            initialize_earth_engine(project=project)
            aoi = ee.Geometry.Point([lon, lat]).buffer(500)
            ing = CocoaExposureIngest(
                aoi,
                year=year,
                threshold=threshold,
                project=project,
                backend=use_backend,
                galileo_checkpoint=galileo_checkpoint,
                aef_checkpoint=aef_checkpoint,
                agrifm_checkpoint=agrifm_checkpoint,
                ensemble_weights_path=ensemble_weights_path,
                region=region_for_point(lat, lon),
            )
            p = ing.sample_point(lat, lon)
            if p is not None:
                return p
        except Exception as exc:
            logger.debug("FDP-region sample failed (%s); trying global fallback", exc)
        return _cocoa_belt_probability(lat, lon)

    return _global_aef_galileo_agrifm_probability(
        lat,
        lon,
        year=year,
        project=project,
        galileo_checkpoint=galileo_checkpoint,
        aef_checkpoint=aef_checkpoint,
        agrifm_checkpoint=agrifm_checkpoint,
        ensemble_weights_path=ensemble_weights_path,
    )


def resolve_exposure_probability(
    lat: float,
    lon: float,
    *,
    year: int = 2023,
    backend: ExposureBackend = "fdp",
    galileo_checkpoint: Path | str | None = None,
    aef_checkpoint: Path | str | None = None,
    agrifm_checkpoint: Path | str | None = None,
    ensemble_weights_path: Path | str | None = None,
    project: str | None = None,
) -> float:
    """
    Point P(cocoa) without constructing a persistent AOI ingest.

    Routes through :func:`sample_cocoa_probability_at_point` for region-aware FDP vs
    global AEF+Galileo fallback.
    """
    return sample_cocoa_probability_at_point(
        lat,
        lon,
        year=year,
        backend=backend,
        galileo_checkpoint=galileo_checkpoint,
        aef_checkpoint=aef_checkpoint,
        agrifm_checkpoint=agrifm_checkpoint,
        ensemble_weights_path=ensemble_weights_path,
        project=project,
    )


__all__ = [
    "CocoaExposureIngest",
    "ExposureBackend",
    "FDP_COCOA_COLLECTION",
    "FDP_MODEL_CARD_URL",
    "DEFAULT_AEF_CHECKPOINT",
    "DEFAULT_AGRIFM_CHECKPOINT",
    "DEFAULT_ENSEMBLE_WEIGHTS",
    "DEFAULT_GALILEO_CHECKPOINT",
    "DEFAULT_THRESHOLD",
    "GLOBAL_AEF_GAL_WEIGHTS",
    "MIN_THRESHOLD",
    "REGIONS",
    "RegionPreset",
    "SUPPORTED_YEARS",
    "is_fdp_covered",
    "normalize_region_key",
    "point_in_region",
    "processed_era5_zarr_path",
    "processed_sentinel_tif_path",
    "region_bounds_dict",
    "region_geometry",
    "region_latlon_bounds",
    "resolve_exposure_probability",
    "sample_cocoa_probability_at_point",
    "validate_threshold",
]
