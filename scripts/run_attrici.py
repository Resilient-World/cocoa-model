#!/usr/bin/env python3
"""
Run ATTRICI counterfactual climate for the cocoa-belt AOI.

Loads factual GSWP3-W5E5 from ``data/raw/gswp3-w5e5/``, subsets to the configured bbox,
invokes :class:`counterfactual.attrici_runner.CounterfactualClimateRunner`, and logs metrics
to MLflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import xarray as xr

# Repo root on path when invoked as ``python scripts/run_attrici.py``
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from counterfactual.attrici_runner import ATTRICIConfig, CounterfactualClimateRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CMIP/ISIMIP short names → file stems under data/raw/gswp3-w5e5/
_W5E5_VARIABLES = (
    "tas",
    "tasmin",
    "tasmax",
    "pr",
    "hurs",
    "rsds",
    "sfcwind",
    "ps",
)

_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tas": ("tas", "t2m", "air_temperature"),
    "tasmin": ("tasmin", "tmin"),
    "tasmax": ("tasmax", "tmax"),
    "pr": ("pr", "precip", "precipitation"),
    "hurs": ("hurs", "rhurs", "relative_humidity"),
    "rsds": ("rsds", "srad", "rsds"),
    "sfcwind": ("sfcwind", "sfcWind", "wind", "ws"),
    "ps": ("ps", "sp", "surface_pressure"),
}


def _find_variable_file(raw_dir: Path, var: str) -> Path | None:
    aliases = _VAR_ALIASES.get(var, (var,))
    for nc in sorted(raw_dir.rglob("*.nc")):
        stem = nc.stem.lower()
        name = nc.name.lower()
        for alias in aliases:
            a = alias.lower()
            if stem == a or f"_{a}_" in name or name.startswith(f"{a}_") or f"_{a}." in name:
                return nc
    return None


def load_factual_w5e5(raw_dir: Path, variables: tuple[str, ...]) -> xr.Dataset:
    """Open GSWP3-W5E5 (or compatible ISIMIP3a) NetCDFs into one Dataset."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Factual data directory not found: {raw_dir}. See docs/data/gswp3_w5e5.md"
        )

    parts: dict[str, xr.DataArray] = {}
    for var in variables:
        path = _find_variable_file(raw_dir, var)
        if path is None:
            logger.warning("No file found for %s under %s", var, raw_dir)
            continue
        ds = xr.open_dataset(path)
        if var in ds:
            da = ds[var]
        else:
            da = ds[list(ds.data_vars)[0]]
        parts[var] = da.rename(var)
        ds.close()

    if not parts:
        raise FileNotFoundError(f"No NetCDF variables loaded from {raw_dir}")
    return xr.Dataset(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ATTRICI counterfactual climate pipeline")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "gswp3-w5e5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "counterfactual",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "attrici_work",
    )
    parser.add_argument(
        "--gmt-file",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "gmt" / "ssa_gmt.nc",
    )
    parser.add_argument(
        "--attrici-venv",
        type=Path,
        default=_REPO_ROOT / ".venv-attrici",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="resilient-cocoa-attrici")
    parser.add_argument("--mlflow-run-name", type=str, default="attrici-counterfactual")
    args = parser.parse_args(argv)

    config = ATTRICIConfig(
        variables=_W5E5_VARIABLES,
        work_dir=args.work_dir,
        counterfactual_dir=args.output_dir,
        gmt_file=args.gmt_file,
        attrici_venv=args.attrici_venv,
    )

    t0 = time.perf_counter()
    factual = load_factual_w5e5(args.raw_dir, config.variables)
    runner = CounterfactualClimateRunner(config)
    runner.prepare_inputs(factual)
    n_lat = factual.sizes.get("lat", factual.sizes.get("latitude", 0))
    n_lon = factual.sizes.get("lon", factual.sizes.get("longitude", 0))
    gridcell_count = int(n_lat) * int(n_lon)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "aoi_bbox": config.aoi_bbox,
                "start_year": config.start_year,
                "end_year": config.end_year,
                "backend": config.backend,
                "n_fourier_modes": config.n_fourier_modes,
                "ssa_window_years": config.ssa_window_years,
                "gridcell_count": gridcell_count,
                "n_variables": len(runner._detrend_variables()),
            }
        )
        out_dir = runner.run()
        elapsed_s = time.perf_counter() - t0
        mlflow.log_metric("runtime_seconds", elapsed_s)
        mlflow.log_metric("gridcell_count", gridcell_count)
        mlflow.log_metric("failed_cell_count", len(runner.failed_cells))
        if runner.failed_cells:
            fail_path = config.work_dir / "failed_cells.jsonl"
            mlflow.log_artifact(str(fail_path))
        mlflow.log_artifact(str(out_dir / "counterfactual_merged.nc"))

    logger.info(
        "Done in %.1f s — %d grid cells, %d failures → %s",
        elapsed_s,
        gridcell_count,
        len(runner.failed_cells),
        out_dir,
    )
    return 0 if not runner.failed_cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
