#!/usr/bin/env python3
"""
Generate CASEJ-emulator training data via Latin-hypercube sampling (West Africa).

Outputs parquet with climate arrays, static vectors, CO2 (ppm), and yield (t/ha).
When RCASEJ is installed, set ``--use-rcase2`` to prefer the Fortran engine (future).

Default: Python CASEJ emulator from Asante et al. (2025) rate equations
(``src/models/casej_process.py`` + ``config/casej/params.yaml``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.casej_process import (
    CASEJSite,
    load_casej_params,
    run_casej_yearly,
    site_to_static_vector,
    synthesize_daily_weather,
    weather_to_climate_tensor,
)

logger = logging.getLogger(__name__)
DEFAULT_OUT = _REPO_ROOT / "data" / "simulations" / "casej_lhs.parquet"
DEFAULT_PARAMS = _REPO_ROOT / "config" / "casej" / "params.yaml"


def _load_lhs_ranges(params_path: Path) -> dict[str, tuple[float, float]]:
    with params_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    lhs = raw.get("lhs_sampling", {})
    return {
        "lat": tuple(lhs.get("lat_range", [4.0, 8.5])),
        "lon": tuple(lhs.get("lon_range", [-8.5, -2.5])),
        "co2": tuple(lhs.get("co2_ppm_range", [380, 700])),
        "precip_scale": tuple(lhs.get("precip_scale_range", [0.6, 1.4])),
        "temp_offset": tuple(lhs.get("temp_offset_range", [-2.0, 2.0])),
        "awc": tuple(lhs.get("awc_mm_range", [80, 220])),
        "slai": tuple(lhs.get("slai_range", [0.0, 3.0])),
        "tree_age": tuple(lhs.get("tree_age_range", [5, 30])),
    }


def generate_lhs(
    n_samples: int,
    *,
    seed: int,
    params_path: Path,
) -> pd.DataFrame:
    ranges = _load_lhs_ranges(params_path)
    dim = 8
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n=n_samples)

    def _scale(u: np.ndarray, key: str) -> np.ndarray:
        lo, hi = ranges[key]
        return lo + u * (hi - lo)

    lats = _scale(unit[:, 0], "lat")
    lons = _scale(unit[:, 1], "lon")
    co2 = _scale(unit[:, 2], "co2")
    precip_scale = _scale(unit[:, 3], "precip_scale")
    temp_offset = _scale(unit[:, 4], "temp_offset")
    awc = _scale(unit[:, 5], "awc")
    slai = _scale(unit[:, 6], "slai")
    tree_age = _scale(unit[:, 7], "tree_age")

    params = load_casej_params(params_path)
    rows: list[dict] = []
    for i in range(n_samples):
        weather = synthesize_daily_weather(
            365,
            seed=seed + i,
            temp_offset_c=float(temp_offset[i]),
            precip_scale=float(precip_scale[i]),
        )
        site = CASEJSite(
            lat=float(lats[i]),
            lon=float(lons[i]),
            awc_mm=float(awc[i]),
            slai=float(slai[i]),
            tree_age_y=float(tree_age[i]),
            co2_ppm=float(co2[i]),
        )
        diag = run_casej_yearly(weather, site, params)
        static = site_to_static_vector(site)
        static[5] = float(np.clip(site.slai / params.slai_max, 0.0, 1.0))
        climate = weather_to_climate_tensor(weather, site.co2_ppm)
        rows.append(
            {
                "sample_id": i,
                "lat": site.lat,
                "lon": site.lon,
                "co2_ppm": site.co2_ppm,
                "slai": site.slai,
                "tree_age_y": site.tree_age_y,
                "awc_mm": site.awc_mm,
                "yield_t_ha": diag["yield_t_ha"],
                "heat_cdd": diag["heat_cdd"],
                "annual_precip_mm": diag["annual_precip_mm"],
                "annual_et_mm": diag["annual_et_mm"],
                "climate": climate.tobytes(),
                "climate_shape": str(climate.shape),
                "static": static.tobytes(),
                "static_shape": str(static.shape),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CASEJ LHS training parquet")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--use-rcase2", action="store_true", help="Reserved: prefer RCASE2 when installed"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.use_rcase2:
        logger.warning("--use-rcase2 not wired; using Python CASEJ emulator")

    df = generate_lhs(args.n_samples, seed=args.seed, params_path=args.params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote %d CASEJ samples to %s", len(df), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
