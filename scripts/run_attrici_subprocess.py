#!/usr/bin/env python3
"""
Invoke ATTRICI v2.x in an isolated venv. GPL-3.0 boundary enforced.

Used when ISIMIP3a counterclim does not cover the variable, grid, or period
(e.g. custom ERA5-Land fields at ~9 km or years after 2019). The main package
never imports ``attrici``; this script shells out to ``.venv-attrici``.

ATTRICI v2.0.1 exposes ``attrici detrend`` (not ``python -m attrici.run``).
Distributions per Mengel et al. 2021 Table 1 are selected from the variable
name inside ATTRICI; :data:`DISTRIBUTIONS` documents that mapping for review.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTRICI_VENV = Path(os.getenv("ATTRICI_VENV", REPO_ROOT / ".venv-attrici"))
DEFAULT_GMT = Path(os.getenv("ATTRICI_GMT_FILE", REPO_ROOT / "data/raw/gmt/ssa_gmt.nc"))

# Mengel et al. 2021, GMD 14, 5269 — Table 1 (§3.2)
DISTRIBUTIONS: dict[str, str] = {
    "tas": "normal",
    "ps": "normal",
    "rlds": "normal",
    "tasrange": "gamma",
    "tasskew": "normal",
    "pr": "bernoulli_gamma",
    "rsds": "normal",
    "sfcwind": "weibull",
    "hurs": "beta",
}

_ATTRICI_VAR_ALIASES: dict[str, str] = {
    "sfcwind": "sfcWind",
}


def _attrici_bin() -> Path:
    return ATTRICI_VENV / "bin" / "attrici"


def _attrici_python() -> Path:
    return ATTRICI_VENV / "bin" / "python"


def _to_attrici_variable(name: str) -> str:
    return _ATTRICI_VAR_ALIASES.get(name, name)


def ensure_venv() -> None:
    """Create ``.venv-attrici`` and install ATTRICI v2.0.1 plus runtime deps."""
    if not ATTRICI_VENV.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(ATTRICI_VENV)])

    pip = ATTRICI_VENV / "bin" / "pip"
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
    # v2.0.1 is tagged on GitHub; PyPI may not publish a wheel.
    subprocess.check_call(
        [
            str(pip),
            "install",
            "attrici @ git+https://github.com/ISI-MIP/attrici@v2.0.1",
            "xarray",
            "netCDF4",
            "scipy",
            "numpy",
            "pandas",
        ]
    )


def run(
    factual_nc: Path,
    out_nc: Path,
    variable: str,
    distribution: str,
    *,
    backend: str = "scipy",
    gmt_file: Path | None = None,
    modes: int = 4,
    ssa_window_years: int = 10,
    overwrite: bool = False,
) -> None:
    """
    Run ATTRICI detrend on a factual NetCDF and write merged counterfactual ``out_nc``.

    Parameters
    ----------
    factual_nc:
        Single-variable (or multi-var) factual climate NetCDF on the target grid.
    out_nc:
        Output path for the merged counterfactual field (``cfact``).
    variable:
        ISIMIP short name (``tas``, ``pr``, …).
    distribution:
        Expected distribution name (validated against :data:`DISTRIBUTIONS`).
    backend:
        ATTRICI solver: ``scipy`` (fast, deterministic) or ``pymc5`` / ``pymc3``.
    gmt_file:
        SSA-smoothed global-mean temperature NetCDF (10-yr window by default).
    modes:
        Number of Fourier modes (ATTRICI default 4).
    ssa_window_years:
        Documented GMT smoothing window (years); must match how ``gmt_file`` was built.
    overwrite:
        Pass ``--overwrite`` to ATTRICI detrend.
    """
    ensure_venv()

    if distribution != DISTRIBUTIONS.get(variable):
        raise ValueError(
            f"Distribution {distribution!r} does not match Table 1 mapping "
            f"{DISTRIBUTIONS.get(variable)!r} for variable {variable!r}"
        )

    factual_nc = factual_nc.resolve()
    if not factual_nc.is_file():
        raise FileNotFoundError(f"Factual NetCDF not found: {factual_nc}")

    gmt = (gmt_file or DEFAULT_GMT).resolve()
    if not gmt.is_file():
        raise FileNotFoundError(
            f"SSA-smoothed GMT file not found: {gmt}\n"
            "Build it with ATTRICI ``ssa`` (10-yr window) or set ATTRICI_GMT_FILE.\n"
            "See docs/data/gswp3_w5e5.md and ``make attrici-env``."
        )

    attrici_var = _to_attrici_variable(variable)
    work_dir = out_nc.parent / f".attrici_work_{variable}"
    detrend_dir = work_dir / "detrend"
    detrend_dir.mkdir(parents=True, exist_ok=True)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ATTRICI detrend variable=%s (%s) solver=%s modes=%d gmt=%s "
        "(GMT SSA window ~%d yr)",
        variable,
        distribution,
        backend,
        modes,
        gmt,
        ssa_window_years,
    )

    detrend_cmd = [
        str(_attrici_bin()),
        "detrend",
        "--gmt-file",
        str(gmt),
        "--input-file",
        str(factual_nc),
        "--output-dir",
        str(detrend_dir),
        "--variable",
        attrici_var,
        "--modes",
        str(modes),
        "--solver",
        backend,
        "--report-variables",
        "cfact",
    ]
    if overwrite:
        detrend_cmd.append("--overwrite")

    subprocess.check_call(detrend_cmd)

    ts_dir = detrend_dir / "timeseries" / attrici_var
    if not ts_dir.is_dir():
        raise FileNotFoundError(
            f"ATTRICI detrend did not produce timeseries at {ts_dir}. "
            "Check factual NetCDF conventions and variable name."
        )

    merge_cmd = [
        str(_attrici_bin()),
        "merge-output",
        str(ts_dir),
        str(out_nc.resolve()),
    ]
    subprocess.check_call(merge_cmd)
    logger.info("Wrote counterfactual %s", out_nc)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run ATTRICI v2 detrend in an isolated GPL venv (full-grid NetCDF)."
    )
    parser.add_argument("--factual", type=Path, required=True, help="Factual input NetCDF")
    parser.add_argument("--out", type=Path, required=True, help="Merged counterfactual NetCDF")
    parser.add_argument(
        "--variable",
        required=True,
        choices=sorted(DISTRIBUTIONS),
        help="Climate variable (distribution fixed by ATTRICI from name)",
    )
    parser.add_argument(
        "--backend",
        default="scipy",
        choices=["scipy", "pymc5", "pymc3"],
        help="Estimator backend (scipy: deterministic, ~10–100× faster than PyMC)",
    )
    parser.add_argument(
        "--gmt",
        type=Path,
        default=None,
        help=f"SSA-smoothed GMT NetCDF (default: {DEFAULT_GMT})",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=4,
        help="Fourier modes (Mengel et al. 2021 default)",
    )
    parser.add_argument(
        "--gmt-smoothing-window",
        type=int,
        default=10,
        dest="ssa_window_years",
        help="Documented GMT SSA window in years (must match --gmt file)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        run(
            args.factual,
            args.out,
            args.variable,
            DISTRIBUTIONS[args.variable],
            backend=args.backend,
            gmt_file=args.gmt,
            modes=args.modes,
            ssa_window_years=args.ssa_window_years,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
