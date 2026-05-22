"""Offline pipeline stubs for DVC repro when ``mock_gee`` is enabled."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ERA5_ZARR = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
DEFAULT_SENTINEL_MANIFEST = _REPO_ROOT / "data" / "processed" / "s2_s1_manifest.json"
DEFAULT_FDP_MANIFEST = _REPO_ROOT / "data" / "raw" / "fdp_ingest_manifest.json"
CASE2_LHS = _REPO_ROOT / "data" / "simulations" / "case2_lhs.parquet"
ALMANAC_LHS = _REPO_ROOT / "data" / "simulations" / "almanac_lhs.parquet"
ERA5_FARM_DIR = _REPO_ROOT / "data" / "era5"


def write_era5_zarr(
    path: Path,
    *,
    start: str = "2020-01-01",
    n_days: int = 365,
) -> Path:
    """Write minimal ERA5-Land-like Zarr for smoke tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = pd.date_range(start, periods=n_days, freq="D")
    lat = np.array([6.0, 7.0], dtype=np.float32)
    lon = np.array([-2.0, -1.0], dtype=np.float32)
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "tmax": (("time", "latitude", "longitude"), np.full(shape, 30.0, dtype=np.float32)),
            "tmin": (("time", "latitude", "longitude"), np.full(shape, 23.0, dtype=np.float32)),
            "tmean": (("time", "latitude", "longitude"), np.full(shape, 26.5, dtype=np.float32)),
            "precip": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype=np.float32)),
            "srad": (("time", "latitude", "longitude"), np.full(shape, 15.0, dtype=np.float32)),
            "vpd_mean": (("time", "latitude", "longitude"), np.full(shape, 1.2, dtype=np.float32)),
            "et0": (("time", "latitude", "longitude"), np.full(shape, 3.5, dtype=np.float32)),
            "sm_root": (("time", "latitude", "longitude"), np.full(shape, 0.28, dtype=np.float32)),
            "wind10m": (("time", "latitude", "longitude"), np.full(shape, 2.0, dtype=np.float32)),
            "rh_mean": (("time", "latitude", "longitude"), np.full(shape, 75.0, dtype=np.float32)),
            "co2_ppm": (("time", "latitude", "longitude"), np.full(shape, 415.0, dtype=np.float32)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def write_sentinel_manifest(path: Path, *, region: str = "ghana") -> Path:
    """Write Sentinel composite stub manifest (no GEE export)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stub": True,
        "region": region,
        "date": date.today().isoformat(),
        "note": "Mock GEE — use sentinel_composite without --stub for production tiles",
        "target_tif": "data/processed/s2_s1.tif",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_fdp_manifest(path: Path, *, region: str = "ghana") -> Path:
    """Write FDP cocoa exposure ingest manifest (no GEE calls)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stub": True,
        "region": region,
        "date": date.today().isoformat(),
        "fdp_asset": "projects/forestdatapartnership/assets/cocoa/model_2025a",
        "backend": "fdp",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_lhs_parquets(
    *,
    n_farms: int = 60,
    seed: int = 42,
    case2_path: Path = CASE2_LHS,
    almanac_path: Path = ALMANAC_LHS,
) -> tuple[Path, Path]:
    """Generate minimal CASE2/ALMANAC LHS tables for yield v2 training."""
    rng = np.random.default_rng(seed)
    ecozones = ["humidforest", "semi_deciduous", "derived_savanna"]
    rows: list[dict[str, object]] = []
    for i in range(n_farms):
        eco = ecozones[i % len(ecozones)]
        lat = 5.5 + rng.uniform(0, 2.0)
        lon = -4.0 + rng.uniform(0, 3.0)
        y_base = 1.2 + rng.normal(0, 0.2)
        rows.append(
            {
                "farm_id": f"farm_{i:04d}",
                "ecozone": eco,
                "planting_density": int(rng.integers(900, 1300)),
                "tree_age": float(rng.uniform(8, 25)),
                "slai": float(rng.uniform(2.0, 5.0)),
                "soil_fc": float(rng.uniform(0.30, 0.42)),
                "soil_wp": float(rng.uniform(0.12, 0.18)),
                "soil_depth": float(rng.uniform(120, 180)),
                "elevation": float(rng.uniform(100, 400)),
                "latitude": lat,
                "longitude": lon,
                "y_case2": y_base * 1000.0,
            }
        )
    case2_df = pd.DataFrame(rows)
    alm_df = case2_df[["farm_id"]].copy()
    alm_df["y_almanac"] = case2_df["y_case2"] * (1.0 + rng.normal(0, 0.05, n_farms))
    case2_path.parent.mkdir(parents=True, exist_ok=True)
    case2_df.to_parquet(case2_path, index=False)
    almanac_path.parent.mkdir(parents=True, exist_ok=True)
    alm_df.to_parquet(almanac_path, index=False)
    return case2_path, almanac_path


def write_per_farm_era5_zarrs(
    lhs_path: Path,
    era5_dir: Path,
    *,
    template_zarr: Path | None = None,
) -> Path:
    """Copy template Zarr per ``farm_id`` for YieldSurrogateV2Dataset."""
    import shutil

    era5_dir = Path(era5_dir)
    era5_dir.mkdir(parents=True, exist_ok=True)
    template = template_zarr or DEFAULT_ERA5_ZARR
    if not template.is_dir():
        write_era5_zarr(template)
    table = pd.read_parquet(lhs_path)
    for farm_id in table["farm_id"].astype(str).unique():
        dest = era5_dir / f"{farm_id}.zarr"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(template, dest)
    return era5_dir


def write_metrics_json(path: Path, metrics: dict[str, float | int | str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
