"""
FTW (Fields of The World) parcel delineation for West African cocoa farms.

Orchestrates the ``ftw-tools`` CLI via :mod:`subprocess` — we do not reimplement FTW
inference. Default model ``3_Class_CCBY_FTW_Pretrained`` is CC-BY licensed for
commercial use; swap to ``FTW_3_Class_FULL_multiWindow`` for non-commercial benchmarking
(both ship in the ftw-baselines v1 release).

Install: ``pip install -e ".[ftw]"``
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import geopandas as gpd

log = structlog.get_logger(__name__)

MODEL_CCBY_COMMERCIAL = "3_Class_CCBY_FTW_Pretrained"
MODEL_FULL_BENCHMARK = "FTW_3_Class_FULL_multiWindow"
SUPPORTED_MODELS: frozenset[str] = frozenset({MODEL_CCBY_COMMERCIAL, MODEL_FULL_BENCHMARK})

YEAR_MIN = 2017
YEAR_MAX = 2025

_INFERENCE_OUTPUT_TIF = "inference_output.tif"
_POLYGONS_PARQUET = "polygons.parquet"
_LULC_FILTERED_PARQUET = "inference_output.parquet"
_FINAL_PARQUET = "cocoa_parcels.geoparquet"


def _bbox_to_ftw_str(bbox: Sequence[float]) -> str:
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (min_lon, min_lat, max_lon, max_lat); got {bbox!r}")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"bbox must satisfy min_lon < max_lon and min_lat < max_lat; got {bbox!r}")
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def _validate_year(year: int) -> None:
    if not YEAR_MIN <= year <= YEAR_MAX:
        raise ValueError(f"year must be in [{YEAR_MIN}, {YEAR_MAX}]; got {year}")


def _run_ftw(cmd: list[str], *, step: str) -> None:
    log.info("FTW %s: %s", step, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            result.stdout,
            f"FTW {step} failed (exit {result.returncode}): {detail}",
        )


def _geopandas():
    import geopandas as gpd

    return gpd


def _enrich_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    projected = gdf.to_crs(gdf.estimate_utm_crs())
    area_m2 = projected.geometry.area
    perimeter_m = projected.geometry.length
    gdf["area_ha"] = area_m2 / 10_000.0
    gdf["compactness"] = (4.0 * math.pi * area_m2) / (perimeter_m**2).where(perimeter_m > 0)
    centroids = gdf.geometry.centroid
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y
    return gdf


def aggregate_to_parcels(
    raster_path: Path | str,
    parcels_gdf: gpd.GeoDataFrame,
    stats: Sequence[str] = ("mean", "std"),
) -> gpd.GeoDataFrame:
    """
      Zonal statistics from a raster onto FTW parcel polygons.

      Used downstream to aggregate ERA5 / Sentinel features to parcel units via
    ``rasterstats.zonal_stats``.
    """
    from rasterstats import zonal_stats

    raster_path = Path(raster_path)
    if not raster_path.is_file():
        raise FileNotFoundError(raster_path)

    stat_list = list(stats)
    if not stat_list:
        raise ValueError("stats must contain at least one statistic name")

    zs = zonal_stats(
        parcels_gdf,
        str(raster_path),
        stats=stat_list,
        geojson_out=False,
    )
    out = parcels_gdf.copy()
    for stat in stat_list:
        out[f"zonal_{stat}"] = [row.get(stat) for row in zs]
    return out


class FTWFieldBoundaries:
    """
    Orchestrate FTW field-boundary inference for a lon/lat bbox and calendar year.

    Parameters
    ----------
    model_name:
        Released FTW model id (default CC-BY commercial model).
    cloud_cover_max, buffer_days, resize_factor:
        Forwarded to ``ftw inference all``.
    ftw_bin:
        FTW CLI executable (default ``ftw`` on ``PATH``).
    """

    def __init__(
        self,
        model_name: str = MODEL_CCBY_COMMERCIAL,
        cloud_cover_max: int = 20,
        buffer_days: int = 14,
        resize_factor: int = 2,
        ftw_bin: str = "ftw",
    ) -> None:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {sorted(SUPPORTED_MODELS)}; got {model_name!r}"
            )
        self.model_name = model_name
        self.cloud_cover_max = cloud_cover_max
        self.buffer_days = buffer_days
        self.resize_factor = resize_factor
        self.ftw_bin = ftw_bin

    def delineate(
        self,
        bbox: Sequence[float],
        year: int,
        out_dir: Path | str,
        overwrite: bool = False,
    ) -> Path:
        """
        Run FTW inference, LULC filter, and cocoa-parcel post-processing.

        Returns
        -------
        Path
            GeoParquet with area, compactness, and centroid columns (fiboa-compatible).
        """
        try:
            import ftw_tools  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ftw-tools is required for FTW delineation. Install with: pip install -e '.[ftw]'"
            ) from exc

        _validate_year(year)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / _FINAL_PARQUET

        if final_path.is_file() and not overwrite:
            log.info("Skipping delineation; exists: %s", final_path)
            return final_path

        self._run_inference_all(bbox, year, out_dir, overwrite=overwrite)
        self._run_filter_by_lulc(out_dir, overwrite=overwrite)

        source_parquet = self._source_parquet_path(out_dir)
        parcels = self.load_parcels(source_parquet)
        parcels.to_parquet(final_path, index=False)
        log.info("Wrote %d cocoa parcels to %s", len(parcels), final_path)
        return final_path

    def load_parcels(
        self,
        parquet_path: Path | str,
        min_ha: float = 0.1,
        max_ha: float = 25.0,
    ) -> gpd.GeoDataFrame:
        """Load FTW polygons and retain plausible cocoa parcel sizes."""
        gpd = _geopandas()
        path = Path(parquet_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        gdf = gpd.read_parquet(path)
        if gdf.empty:
            return _enrich_parcels(gdf)

        gdf = _enrich_parcels(gdf)
        mask = (gdf["area_ha"] >= min_ha) & (gdf["area_ha"] <= max_ha)
        filtered = gdf.loc[mask].copy()
        log.info(
            "Parcels %s: %d -> %d (%.1f–%.1f ha)",
            path.name,
            len(gdf),
            len(filtered),
            min_ha,
            max_ha,
        )
        return filtered

    def _source_parquet_path(self, out_dir: Path) -> Path:
        """Prefer LULC-filtered polygons when step 2 completed."""
        lulc = out_dir / _LULC_FILTERED_PARQUET
        if lulc.is_file():
            return lulc
        return out_dir / _POLYGONS_PARQUET

    def _run_inference_all(
        self,
        bbox: Sequence[float],
        year: int,
        out_dir: Path,
        *,
        overwrite: bool,
    ) -> None:
        polygons = out_dir / _POLYGONS_PARQUET
        inference_tif = out_dir / _INFERENCE_OUTPUT_TIF
        if polygons.is_file() and inference_tif.is_file() and not overwrite:
            log.info("Skipping ftw inference all; outputs exist in %s", out_dir)
            return

        cmd = [
            self.ftw_bin,
            "inference",
            "all",
            "--bbox",
            _bbox_to_ftw_str(bbox),
            "--year",
            str(year),
            "--out",
            str(out_dir),
            "--model",
            self.model_name,
            "--cloud_cover_max",
            str(self.cloud_cover_max),
            "--buffer_days",
            str(self.buffer_days),
            "--resize_factor",
            str(self.resize_factor),
        ]
        if overwrite:
            cmd.append("--overwrite")
        _run_ftw(cmd, step="inference all")

    def _run_filter_by_lulc(self, out_dir: Path, *, overwrite: bool) -> None:
        inference_tif = out_dir / _INFERENCE_OUTPUT_TIF
        if not inference_tif.is_file():
            raise FileNotFoundError(
                f"Expected {inference_tif} after ftw inference all; run delineate with overwrite=True"
            )

        lulc_parquet = out_dir / _LULC_FILTERED_PARQUET
        if lulc_parquet.is_file() and not overwrite:
            log.info("Skipping ftw inference filter-by-lulc; exists: %s", lulc_parquet)
            return

        cmd = [
            self.ftw_bin,
            "inference",
            "filter-by-lulc",
            str(inference_tif),
            "--collection_name",
            "esa-worldcover",
            "-o",
            str(lulc_parquet),
        ]
        if overwrite:
            cmd.append("--overwrite")
        _run_ftw(cmd, step="filter-by-lulc")
