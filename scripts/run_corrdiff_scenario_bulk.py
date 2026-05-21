#!/usr/bin/env python3
"""
Pre-compute CorrDiff-CMIP6 scenario Zarr caches for all 48 (SSP × horizon × region) strata.

HPC-only (~4 h/stratum on H100). See docs/corrdiff_compute.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from counterfactual.corrdiff_downscaler import (
    DEFAULT_OUTPUT_VARIABLES,
    CorrDiffCMIP6Downscaler,
    corrdiff_cache_path,
)
from data.cocoa_exposure import REGIONS

logger = logging.getLogger(__name__)

SCENARIOS = ("ssp245", "ssp585")
HORIZONS = (2030, 2050, 2080)


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _parse_strata(raw: list[str]) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected scenario:horizon:region, got {item!r}")
        out.append((parts[0], int(parts[1]), parts[2]))
    return out


def _all_strata() -> list[tuple[str, int, str]]:
    return [
        (scenario, horizon, region)
        for scenario in SCENARIOS
        for horizon in HORIZONS
        for region in sorted(REGIONS.keys())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk CorrDiff-CMIP6 scenario downscaling")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "processed",
    )
    parser.add_argument("--strata", nargs="*", help="Subset e.g. ssp245:2030:ghana")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--number-of-samples", type=int, default=None)
    parser.add_argument("--era5-zarr", type=Path, default=None)
    parser.add_argument("--cmip6-zarr", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    strata = _parse_strata(args.strata) if args.strata else _all_strata()
    manifest_path = args.processed_dir / "corrdiff_bulk_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    completed = manifest.get("completed", {})
    t0_all = time.perf_counter()
    elapsed_strata: list[float] = []

    for i, (scenario, horizon, region) in enumerate(strata):
        cache = corrdiff_cache_path(args.processed_dir, scenario, horizon, region)
        if cache.is_dir() and not args.force:
            logger.info("Skip existing %s", cache.name)
            continue
        if args.dry_run:
            logger.info("Would build %s", cache)
            continue

        n_samples = args.number_of_samples or 8
        t0 = time.perf_counter()
        downscaler = CorrDiffCMIP6Downscaler(
            experiment_id=scenario,  # type: ignore[arg-type]
            number_of_samples=n_samples,
            region=region,
            historical_zarr_path=args.era5_zarr,
            cmip6_zarr_path=args.cmip6_zarr,
        )
        downscaler.downscale_horizon_year(horizon, list(DEFAULT_OUTPUT_VARIABLES))
        downscaler.to_zarr(cache)
        dt = time.perf_counter() - t0
        elapsed_strata.append(dt)
        key = f"{scenario}:{horizon}:{region}"
        completed[key] = {
            "path": str(cache.relative_to(_REPO_ROOT)),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "seconds": round(dt, 1),
            "number_of_samples": n_samples,
        }
        eta = (len(strata) - i - 1) * (sum(elapsed_strata) / len(elapsed_strata))
        logger.info("Done %s in %.1fs (ETA %.0fs)", key, dt, eta)

    if not args.dry_run:
        manifest["completed"] = completed
        manifest["git_hash"] = _git_hash()
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        args.processed_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Wrote manifest %s (total %.1fs)", manifest_path, time.perf_counter() - t0_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
