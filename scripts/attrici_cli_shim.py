#!/usr/bin/env python3
"""
Shim between the conceptual per-cell ``attrici run`` API and ATTRICI v2.0.1 ``detrend``.

ATTRICI v2 selects distribution/link from the variable name (Mengel et al. 2021 Table 1);
they are not exposed on the CLI. This script forwards to ``attrici detrend`` with the
flags that v2.0.1 actually supports.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_detrend_argv(
    *,
    attrici_bin: Path,
    gmt_file: Path,
    input_file: Path,
    output_dir: Path,
    variable: str,
    lat: float,
    lon: float,
    modes: int,
    solver: str,
    start_date: str | None,
    stop_date: str | None,
    overwrite: bool,
) -> list[str]:
    cmd = [
        str(attrici_bin),
        "detrend",
        "--gmt-file",
        str(gmt_file),
        "--input-file",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--variable",
        variable,
        "--cells",
        f"{lat:g},{lon:g}",
        "--modes",
        str(modes),
        "--solver",
        solver,
        "--report-variables",
        "cfact",
    ]
    if start_date is not None:
        cmd.extend(["--start-date", start_date])
    if stop_date is not None:
        cmd.extend(["--stop-date", stop_date])
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Invoke ATTRICI detrend for one grid cell (v2.0.1 CLI shim)."
    )
    parser.add_argument("--attrici-bin", type=Path, required=True)
    parser.add_argument("--gmt-file", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--solver", type=str, default="scipy", choices=["scipy", "pymc5", "pymc3"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--stop-date", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cmd = build_detrend_argv(
        attrici_bin=args.attrici_bin,
        gmt_file=args.gmt_file,
        input_file=args.input_file,
        output_dir=args.output_dir,
        variable=args.variable,
        lat=args.lat,
        lon=args.lon,
        modes=args.modes,
        solver=args.solver,
        start_date=args.start_date,
        stop_date=args.stop_date,
        overwrite=args.overwrite,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
