#!/usr/bin/env python3
"""
Hindcast validation: CorrDiff ensemble CRPS vs linear-delta MAE (2021–2024 ERA5 hold-out).

Gate (default): CorrDiff CRPS <= linear-delta MAE on tmean, precip, srad, rh_mean.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from counterfactual.cmip6_scenarios import ScenarioBuilder
from counterfactual.corrdiff_downscaler import (
    CorrDiffCMIP6Downscaler,
    corrdiff_cache_path,
    write_synthetic_corrdiff_cache,
)
from data.cocoa_exposure import REGIONS, normalize_region_key

logger = logging.getLogger(__name__)

CORE_CHANNELS = ("tmean", "precip", "srad", "rh_mean")
HINDCAST_START = 2021
HINDCAST_END = 2024


def _crps(obs: np.ndarray, ensemble: np.ndarray) -> float:
    """CRPS for 1D obs vs ensemble members (same length)."""
    ens = np.asarray(ensemble, dtype=np.float64).ravel()
    y = float(np.asarray(obs, dtype=np.float64).ravel()[0])
    term1 = np.mean(np.abs(ens - y))
    term2 = 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]))
    return float(term1 - term2)


def _mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(obs) - np.asarray(pred))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-zarr", type=Path, default=_REPO_ROOT / "data/processed/era5_2020_2024.zarr")
    parser.add_argument("--cmip6-zarr", type=Path, default=_REPO_ROOT / "data/processed/cmip6_ensemble.zarr")
    parser.add_argument("--processed-dir", type=Path, default=_REPO_ROOT / "data/processed")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument("--region", default="ghana")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--reports-dir", type=Path, default=_REPO_ROOT / "reports/scenarios")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    region = normalize_region_key(args.region)
    horizon = HINDCAST_END if args.quick else HINDCAST_END
    window = (f"{HINDCAST_START}-01-01", f"{HINDCAST_END}-12-31")

    if not args.era5_zarr.is_dir():
        logger.error("ERA5 Zarr missing at %s", args.era5_zarr)
        return 1

    era5 = xr.open_zarr(args.era5_zarr, consolidated=True)
    preset = REGIONS[region]
    era5_pt = era5.sel(
        time=slice(str(HINDCAST_START), str(HINDCAST_END)),
        lat=slice(preset.south, preset.north),
        lon=slice(preset.west, preset.east),
        method="nearest",
    )
    if args.quick:
        era5_pt = era5_pt.isel(time=slice(0, 14))

    linear_ds = None
    if args.cmip6_zarr.is_dir():
        builder = ScenarioBuilder(str(args.era5_zarr), str(args.cmip6_zarr))
        linear_ds = builder.build_scenario(args.scenario, window).sel(
            lat=slice(preset.south, preset.north),
            lon=slice(preset.west, preset.east),
            method="nearest",
        )
        if args.quick:
            linear_ds = linear_ds.isel(time=slice(0, 14))

    cache = corrdiff_cache_path(args.processed_dir, args.scenario, horizon, region)
    if not cache.is_dir():
        if args.quick:
            write_synthetic_corrdiff_cache(cache, scenario=args.scenario, horizon=horizon, region=region)
        else:
            downscaler = CorrDiffCMIP6Downscaler(
                experiment_id=args.scenario,  # type: ignore[arg-type]
                region=region,
                historical_zarr_path=args.era5_zarr,
                cmip6_zarr_path=args.cmip6_zarr,
            )
            downscaler.downscale_horizon_year(horizon, list(CORE_CHANNELS))
            downscaler.to_zarr(cache)

    corrdiff_ds = xr.open_zarr(cache, consolidated=True)
    if args.quick:
        corrdiff_ds = corrdiff_ds.isel(time=slice(0, 14))

    rows: list[str] = []
    passed = True
    for ch in CORE_CHANNELS:
        if ch not in era5_pt:
            logger.warning("Skip %s (not in ERA5)", ch)
            continue
        obs = era5_pt[ch].mean(dim=("lat", "lon")).values
        if "sample" in corrdiff_ds.dims and ch in corrdiff_ds:
            ens = corrdiff_ds[ch].mean(dim=("lat", "lon")).values
            crps_vals = [_crps(obs[t], ens[t, :]) for t in range(obs.shape[0])]
            crps_mean = float(np.mean(crps_vals))
        else:
            crps_mean = float("nan")
            passed = False

        if linear_ds is not None and ch in linear_ds:
            pred = linear_ds[ch].mean(dim=("lat", "lon")).values
            mae_mean = _mae(obs, pred)
        else:
            mae_mean = float("nan")
            passed = False

        ok = crps_mean <= mae_mean + args.tolerance
        passed = passed and ok
        rows.append(f"| {ch} | {crps_mean:.4f} | {mae_mean:.4f} | {'pass' if ok else 'FAIL'} |")

    report_date = date.today().isoformat()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"corrdiff_validation_{report_date}.md"
    body = "\n".join(
        [
            f"# CorrDiff vs linear-delta validation ({report_date})",
            "",
            f"- scenario: `{args.scenario}`",
            f"- region: `{region}`",
            f"- quick: `{args.quick}`",
            "",
            "| channel | CorrDiff CRPS | linear MAE | gate |",
            "|---------|---------------|------------|------|",
            *rows,
            "",
            f"**Overall:** {'PASS' if passed else 'FAIL'}",
        ]
    )
    report_path.write_text(body, encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
