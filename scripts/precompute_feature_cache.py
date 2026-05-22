#!/usr/bin/env python3
"""
Pre-resolve API features on a regular lat/lon grid and write ``features_cache.zarr``.

Covers cocoa-producing regions (Ghana, Côte d'Ivoire, Cameroon, Nigeria, Indonesia)
at 0.05° by default. Climate stacks are taken from ``ERA5_ZARR_PATH`` when present;
static covariates are sampled via :class:`~api.feature_resolver.FarmFeatureResolver`
(GEE: SoilGrids, SRTM, CHIRPS, WDPA, FDP cocoa).

ISRIC SoilGrids source: https://files.isric.org/soilgrids/latest/data/

Example::

    python scripts/precompute_feature_cache.py --step 0.05 --years 2023
    python scripts/precompute_feature_cache.py --regions gha,civ --max-points 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from api.feature_resolver import (
    DEFAULT_ERA5_ZARR,
    DEFAULT_FEATURES_CACHE_ZARR,
    RESOLVED_STATIC_NAMES,
    SEQUENCE_LENGTH,
    FarmFeatureResolver,
    FeatureResolverConfig,
    _climate_tensor_from_dataset,
    _lat_lon_coord_names,
    round_to_grid,
)
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES, N_CLIMATE_CHANNELS

logger = logging.getLogger(__name__)

from data.cocoa_exposure import REGIONS, normalize_region_key, region_latlon_bounds

# lat_min, lat_max, lon_min, lon_max — aligned with data.cocoa_exposure.REGIONS
REGION_BBOXES: dict[str, tuple[float, float, float, float]] = {
    key: region_latlon_bounds(key) for key in REGIONS
}
# CLI short aliases
REGION_BBOXES.update(
    {
        "gha": REGION_BBOXES["ghana"],
        "civ": REGION_BBOXES["civ"],
        "cmr": REGION_BBOXES["cameroon"],
        "nga": REGION_BBOXES["nigeria"],
        "idn": REGION_BBOXES["indonesia"],
        "ecu": REGION_BBOXES["ecuador"],
        "per": REGION_BBOXES["peru"],
        "col": REGION_BBOXES["colombia"],
    }
)


def grid_cells(
    bbox: tuple[float, float, float, float],
    *,
    step: float,
) -> list[tuple[float, float]]:
    lat_min, lat_max, lon_min, lon_max = bbox
    lats = np.arange(lat_min, lat_max + step * 0.5, step)
    lons = np.arange(lon_min, lon_max + step * 0.5, step)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def subsample_cells(
    cells: list[tuple[float, float]], max_points: int, seed: int
) -> list[tuple[float, float]]:
    if len(cells) <= max_points:
        return cells
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cells), size=max_points, replace=False)
    return [cells[i] for i in sorted(idx)]


def _climate_from_era5_zarr(
    era5_path: Path,
    lat: float,
    lon: float,
    year: int,
) -> np.ndarray:
    ds = xr.open_zarr(era5_path, consolidated=True)
    lat_name, lon_name = _lat_lon_coord_names(ds)
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    return _climate_tensor_from_dataset(point, year)


def build_cache_dataset(
    cells: list[tuple[float, float]],
    *,
    years: list[int],
    resolver: FarmFeatureResolver,
    era5_path: Path | None,
) -> xr.Dataset:
    lats = sorted({round_to_grid(lat, lon)[0] for lat, lon in cells})
    lons = sorted({round_to_grid(lat, lon)[1] for lat, lon in cells})
    n_lat, n_lon = len(lats), len(lons)
    lat_index = {v: i for i, v in enumerate(lats)}
    lon_index = {v: i for i, v in enumerate(lons)}

    static_arrays = {
        name: np.full((n_lat, n_lon), np.nan, dtype=np.float32) for name in RESOLVED_STATIC_NAMES
    }
    model_static = np.full((n_lat, n_lon, 13), np.nan, dtype=np.float32)
    climates = {
        year: np.full((n_lat, n_lon, SEQUENCE_LENGTH, N_CLIMATE_CHANNELS), np.nan, dtype=np.float32)
        for year in years
    }

    for i, (lat, lon) in enumerate(cells):
        lat_r, lon_r = round_to_grid(lat, lon, resolver.config.grid_step_deg)
        li, lj = lat_index[lat_r], lon_index[lon_r]
        if i % 50 == 0:
            logger.info("Grid %d/%d (%.2f, %.2f)", i + 1, len(cells), lat_r, lon_r)

        resolved = resolver._resolve_static_from_gee(lat_r, lon_r, year=years[0])
        for name, val in zip(RESOLVED_STATIC_NAMES, resolved.as_vector(), strict=True):
            static_arrays[name][li, lj] = val
        model_static[li, lj, :] = resolver._pack_model_static_vector(resolved, lat_r, lon_r)

        for year in years:
            if era5_path is not None and era5_path.is_dir():
                try:
                    climates[year][li, lj, :, :] = _climate_from_era5_zarr(
                        era5_path, lat_r, lon_r, year
                    )
                    continue
                except Exception as exc:
                    logger.debug("ERA5 zarr miss (%s, %s, %d): %s", lat_r, lon_r, year, exc)
            tensor = resolver.resolve_climate(lat_r, lon_r, year)
            climates[year][li, lj, :, :] = tensor.squeeze(0).numpy()

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name, arr in static_arrays.items():
        data_vars[name] = (("latitude", "longitude"), arr)
    data_vars["model_static"] = (("latitude", "longitude", "static_idx"), model_static)

    if len(years) == 1:
        y0 = years[0]
        data_vars["climate"] = (
            ("latitude", "longitude", "day", "channel"),
            climates[y0],
        )
    else:
        stack = np.stack([climates[y] for y in years], axis=0)
        data_vars["climate"] = (
            ("year", "latitude", "longitude", "day", "channel"),
            stack,
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "latitude": lats,
            "longitude": lons,
            "day": np.arange(SEQUENCE_LENGTH),
            "channel": list(CLIMATE_CHANNEL_NAMES),
            "static_idx": np.arange(13),
            **({"year": years} if len(years) > 1 else {}),
        },
        attrs={
            "grid_step_deg": resolver.config.grid_step_deg,
            "isric_url": "https://files.isric.org/soilgrids/latest/data/",
        },
    )
    return ds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute features_cache.zarr")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FEATURES_CACHE_ZARR,
        help="Output Zarr path",
    )
    parser.add_argument("--era5-zarr", type=Path, default=DEFAULT_ERA5_ZARR)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--regions", type=str, default="gha,civ,cmr,nga,idn")
    parser.add_argument("--years", type=str, default="2023", help="Comma-separated years")
    parser.add_argument("--max-points", type=int, default=2000, help="Subsample cap per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gee-project", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    region_keys = [r.strip().lower() for r in args.regions.split(",") if r.strip()]
    cells: list[tuple[float, float]] = []
    for key in region_keys:
        try:
            norm = normalize_region_key(key)
        except KeyError:
            if key not in REGION_BBOXES:
                parser.error(f"Unknown region {key!r}; choose from {sorted(REGIONS)}")
            norm = key
        cells.extend(grid_cells(REGION_BBOXES.get(key, REGION_BBOXES[norm]), step=args.step))

    unique = list({round_to_grid(lat, lon, args.step): (lat, lon) for lat, lon in cells}.values())
    cells = subsample_cells(unique, args.max_points, args.seed)
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    resolver = FarmFeatureResolver(
        FeatureResolverConfig(
            era5_zarr_path=args.era5_zarr,
            features_cache_zarr_path=args.out,
            grid_step_deg=args.step,
            use_real_features=True,
            gee_project=args.gee_project,
        )
    )

    era5 = args.era5_zarr if args.era5_zarr.is_dir() else None
    if era5 is None:
        logger.warning(
            "ERA5 Zarr not found at %s; climate will use GEE/mock per cell", args.era5_zarr
        )

    ds = build_cache_dataset(cells, years=years, resolver=resolver, era5_path=era5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.out, mode="w", consolidated=True)
    logger.info("Wrote %s (%d lat × %d lon)", args.out, ds.sizes["latitude"], ds.sizes["longitude"])

    index = pd.DataFrame([{"lat": lat, "lon": lon} for lat, lon in cells])
    index_path = args.out.parent / "features_cache_index.parquet"
    index.to_parquet(index_path, index=False)
    logger.info("Wrote cell index %s", index_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
