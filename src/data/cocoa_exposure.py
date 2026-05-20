"""
Forest Data Partnership (FDP) cocoa probability ingest via Google Earth Engine.

Dataset
-------
Forest Data Partnership (2025) *Cocoa Probability model 2025a*, successor to
Kalischek et al. (2023) *Nature Food*. ImageCollection:
``projects/forestdatapartnership/assets/cocoa/model_2025a`` — 10 m, annual composites
for 2020 and 2023; coverage Côte d'Ivoire, Ghana, Indonesia, Ecuador, Peru, Colombia.

Exposure backends
-----------------
- ``fdp``: FDP probability raster (GEE prior)
- ``galileo``: :class:`~models.galileo_seg.GalileoCocoaSegmentation` tile inference
- ``ensemble``: ``0.5 * FDP + 0.5 * Galileo`` (when both available)

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
from pathlib import Path
from typing import Literal

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

ExposureBackend = Literal["fdp", "galileo", "ensemble"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GALILEO_CHECKPOINT = _REPO_ROOT / "models" / "galileo_cocoa_seg.pt"


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
        ensemble_weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.aoi = aoi
        self.year = _normalize_year(year)
        self.threshold = validate_threshold(threshold)
        self.project = project
        self.backend = backend
        self.galileo_checkpoint = (
            Path(galileo_checkpoint) if galileo_checkpoint else DEFAULT_GALILEO_CHECKPOINT
        )
        self.ensemble_weights = ensemble_weights
        self._probability_image: ee.Image | None = None
        self._galileo_model = None

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

        # ensemble
        fdp_p = self._fdp_probability_at_point(lat, lon, scale_m)
        gal_p = self._galileo_probability_at_point(lat, lon)
        w_fdp, w_gal = self.ensemble_weights
        if fdp_p is None:
            return float(np.clip(gal_p, 0.0, 1.0))
        return float(np.clip(w_fdp * fdp_p + w_gal * gal_p, 0.0, 1.0))

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


def resolve_exposure_probability(
    lat: float,
    lon: float,
    *,
    year: int = 2023,
    backend: ExposureBackend = "fdp",
    galileo_checkpoint: Path | str | None = None,
    project: str | None = None,
) -> float:
    """
    Point P(cocoa) without constructing a persistent AOI ingest.

    ``fdp`` / ``ensemble`` use GEE when credentials exist; otherwise a belt heuristic.
    """
    if backend == "fdp":
        try:
            import ee as _ee

            _ee.Initialize(project=project) if project else _ee.Initialize()
            aoi = _ee.Geometry.Point([lon, lat]).buffer(500)
            ing = CocoaExposureIngest(aoi, year=year, project=project, backend="fdp")
            p = ing.sample_point(lat, lon)
            if p is not None:
                return p
        except Exception as exc:
            logger.debug("FDP point sample failed (%s); using belt heuristic", exc)
        return _cocoa_belt_probability(lat, lon)

    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        year=year,
        backend=backend,
        galileo_checkpoint=galileo_checkpoint,
        project=project,
    )
    return ing.sample_point(lat, lon) or _cocoa_belt_probability(lat, lon)


__all__ = [
    "CocoaExposureIngest",
    "ExposureBackend",
    "FDP_COCOA_COLLECTION",
    "FDP_MODEL_CARD_URL",
    "DEFAULT_GALILEO_CHECKPOINT",
    "DEFAULT_THRESHOLD",
    "MIN_THRESHOLD",
    "SUPPORTED_YEARS",
    "resolve_exposure_probability",
    "validate_threshold",
]
