#!/usr/bin/env python3
"""
Pre-compute ATTRICI v2.0.1 counterfactual ERA5 for Côte d'Ivoire + Ghana (1980–2024).

Writes ``data/processed/era5_counterfactual/civ_ghana_1980_2024.zarr`` via subprocess-only
:class:`data.attrici_counterfactual.ATTRICICounterfactual` (GPL boundary).

Example::

    python scripts/run_attrici_civ_ghana.py \\
        --factual-zarr data/processed/era5_2020_2024.zarr \\
        --gmt-file data/raw/gmt/ssa_gmt.nc
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.attrici_counterfactual import (
    ATTRICICounterfactual,
    ERA5_VARIABLES,
    RegionBounds,
    TimeRange,
)

logger = logging.getLogger(__name__)

DEFAULT_FACTUAL = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
DEFAULT_GMT = _REPO_ROOT / "data" / "raw" / "gmt" / "ssa_gmt.nc"
DEFAULT_OUT = _REPO_ROOT / "data" / "processed" / "era5_counterfactual" / "civ_ghana_1980_2024.zarr"

# Cocoa belt West Africa (CIV + Ghana)
CIV_GHA_REGION = RegionBounds(lat_min=4.0, lat_max=8.5, lon_min=-8.5, lon_max=-2.5)
CIV_GHA_YEARS = TimeRange(start_year=1980, end_year=2024)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ATTRICI counterfactual ERA5 for Côte d'Ivoire + Ghana"
    )
    parser.add_argument("--factual-zarr", type=Path, default=DEFAULT_FACTUAL)
    parser.add_argument("--gmt-file", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--out-zarr", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--attrici-bin", type=str, default=os.environ.get("ATTRICI_BIN", "attrici"))
    parser.add_argument("--backend", choices=("scipy", "pymc5"), default="scipy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--variables",
        type=str,
        default=",".join(ERA5_VARIABLES),
        help="Comma-separated ERA5 variables (default: full CASEJ set)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    variables = tuple(v.strip() for v in args.variables.split(",") if v.strip())
    model = ATTRICICounterfactual(
        args.factual_zarr,
        gmt_file=args.gmt_file,
        cache_dir=_REPO_ROOT / "data" / "cache" / "attrici_counterfactual",
        attrici_bin=args.attrici_bin,
        backend=args.backend,
    )
    out = model.build_counterfactual_zarr(
        variables,
        region=CIV_GHA_REGION,
        time_range=CIV_GHA_YEARS,
        output_zarr=args.out_zarr,
        overwrite=args.overwrite,
    )
    logger.info("Counterfactual store: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
