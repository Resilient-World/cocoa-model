#!/usr/bin/env python3
"""
Bootstrap online conformal state for /simulate-scenario strata.

Runs synthetic farm calls per (region, scenario, horizon) with CMIP6-style score noise,
then writes ``data/processed/conformal_initial_state.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from api.feature_resolver import climate_tensor_from_dataset_point
from api.online_conformal_store import stratum_key
from counterfactual.cmip6_scenarios import ScenarioBuilder
from data.cocoa_exposure import REGIONS, region_latlon_bounds
from data.yield_panel import load_icco_tables
from models.cqr import ConformalCalibrator, load_quantile_yield_model
from models.eci import ECIIntegral

logger = logging.getLogger(__name__)

SCENARIOS = ("ssp245", "ssp585")
HORIZONS = (2030, 2050, 2080)

_REGION_MEAN_YIELD: dict[str, float] = {
    "ghana": 0.55,
    "civ": 0.58,
    "cameroon": 0.52,
    "nigeria": 0.50,
    "indonesia": 0.48,
    "ecuador": 0.54,
    "peru": 0.51,
    "colombia": 0.53,
}


def _regional_mean_yields() -> dict[str, float]:
    out = dict(_REGION_MEAN_YIELD)
    try:
        df = load_icco_tables()
        iso_map = {
            "GHA": "ghana",
            "CIV": "civ",
            "CMR": "cameroon",
            "NGA": "nigeria",
            "ECU": "ecuador",
            "IDN": "indonesia",
        }
        for iso, key in iso_map.items():
            sub = df[df["country_iso3"] == iso]
            if len(sub):
                out[key] = float((sub["production_tonnes"] / sub["planted_area_ha"]).mean())
    except FileNotFoundError:
        logger.warning("ICCO tables missing; using default regional yield priors")
    return out


def _synthetic_climate_tensor(rng: np.random.Generator, length: int = 365) -> torch.Tensor:
    seasonal = np.sin(2 * np.pi * np.arange(length) / 365.0)
    tmax = 30.0 + 2.0 * seasonal + rng.normal(0, 0.3, length)
    tmin = tmax - 7.0
    from models.yield_surrogate import N_CLIMATE_CHANNELS

    climate = np.zeros((length, N_CLIMATE_CHANNELS), dtype=np.float32)
    climate[:, 0] = tmax
    climate[:, 1] = tmin
    climate[:, 2] = 0.5 * (tmax + tmin)
    climate[:, 3] = np.clip(rng.gamma(2, 3, length), 0, 50)
    climate[:, 4] = 15.0 + 2.0 * seasonal
    climate[:, 5] = 1.2
    climate[:, 6] = 3.5
    climate[:, 7] = 0.28
    climate[:, 8] = 2.0
    climate[:, 9] = 75.0
    climate[:, 10] = 415.0
    return torch.from_numpy(climate).unsqueeze(0)


def _encode_static_simple(current_yield: float) -> torch.Tensor:
    from models.yield_surrogate import N_STATIC_SITE

    static = torch.zeros(1, N_STATIC_SITE, dtype=torch.float32)
    static[0, 2] = current_yield / 5.0
    return static


def _sample_latlon(region: str, rng: np.random.Generator) -> tuple[float, float]:
    lat_min, lat_max, lon_min, lon_max = region_latlon_bounds(region)
    lat = float(rng.uniform(lat_min, lat_max))
    lon = float(rng.uniform(lon_min, lon_max))
    return lat, lon


def calibrate_stratum(
    *,
    region: str,
    scenario: str,
    horizon: int,
    n_calls: int,
    cqr_model: torch.nn.Module,
    era5_zarr: Path | None,
    cmip6_zarr: Path | None,
    regional_yields: dict[str, float],
    alpha: float,
    eta: float,
    seed: int,
) -> dict[str, float]:
    key = stratum_key(scenario, horizon, region)
    rng = np.random.default_rng(seed)
    updater = ECIIntegral(alpha, eta=eta, decay=0.95, window=100, q_init=0.0)
    mean_y = regional_yields.get(region, 0.55)
    builder = None
    if era5_zarr and cmip6_zarr and era5_zarr.is_dir() and cmip6_zarr.is_dir():
        builder = ScenarioBuilder(str(era5_zarr), str(cmip6_zarr))

    for i in range(n_calls):
        lat, lon = _sample_latlon(region, rng)
        if builder is not None:
            try:
                window = (f"{horizon}-01-01", f"{horizon}-12-31")
                ds = builder.build_scenario(scenario, window)
                climate = climate_tensor_from_dataset_point(ds, lat, lon, 2023)
            except Exception:
                climate = _synthetic_climate_tensor(rng)
        else:
            climate = _synthetic_climate_tensor(rng)
            # CMIP6 ensemble spread proxy
            climate = climate + torch.tensor(
                rng.normal(0, 0.05, climate.shape), dtype=torch.float32
            )

        static = _encode_static_simple(mean_y + rng.normal(0, 0.08))
        cqr_model.eval()
        with torch.no_grad():
            q_pred = cqr_model(climate, static).cpu().numpy()
        q_lo, q_hi = float(q_pred[0, 0]), float(q_pred[0, 2])
        observed_y = float(mean_y + rng.normal(0, 0.12))
        q_adj = updater.current_threshold
        adj_lo, adj_hi = q_lo - q_adj, q_hi + q_adj
        score = float(
            ConformalCalibrator.conformity_scores(
                np.array([observed_y]),
                np.array([adj_lo]),
                np.array([adj_hi]),
            )[0]
        )
        covered = adj_lo <= observed_y <= adj_hi
        updater.update(score, covered=covered)

    return {
        "q_t": float(updater.current_threshold),
        "q_init": float(updater.current_threshold),
        "n_calls": n_calls,
        "eta": eta,
        "alpha": alpha,
        "region": region,
        "scenario": scenario,
        "horizon_year": horizon,
        "key": key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap online conformal initial state")
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "data/processed/conformal_initial_state.json"
    )
    parser.add_argument("--cqr-checkpoint", type=Path, default=_REPO_ROOT / "models/cqr_yield.pt")
    parser.add_argument(
        "--era5-zarr", type=Path, default=_REPO_ROOT / "data/processed/era5_2020_2024.zarr"
    )
    parser.add_argument(
        "--cmip6-zarr", type=Path, default=_REPO_ROOT / "data/processed/cmip6_ensemble.zarr"
    )
    parser.add_argument("--n-calls", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="100 calls per stratum")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    n_calls = 100 if args.quick else args.n_calls

    if not args.cqr_checkpoint.is_file():
        logger.error("CQR checkpoint not found: %s", args.cqr_checkpoint)
        raise SystemExit(1)

    cqr_model = load_quantile_yield_model(args.cqr_checkpoint, galileo_dim=0)
    regional_yields = _regional_mean_yields()
    era5 = args.era5_zarr if args.era5_zarr.is_dir() else None
    cmip6 = args.cmip6_zarr if args.cmip6_zarr.is_dir() else None

    blob: dict[str, dict[str, float]] = {}
    seed = args.seed
    for region in REGIONS:
        for scenario in SCENARIOS:
            for horizon in HORIZONS:
                entry = calibrate_stratum(
                    region=region,
                    scenario=scenario,
                    horizon=horizon,
                    n_calls=n_calls,
                    cqr_model=cqr_model,
                    era5_zarr=era5,
                    cmip6_zarr=cmip6,
                    regional_yields=regional_yields,
                    alpha=args.alpha,
                    eta=args.eta,
                    seed=seed,
                )
                blob[entry["key"]] = entry
                seed += 1
                logger.info(
                    "Calibrated %s q_t=%.4f (%d calls)",
                    entry["key"],
                    entry["q_t"],
                    n_calls,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    logger.info("Wrote %s (%d strata)", args.out, len(blob))


if __name__ == "__main__":
    main()
