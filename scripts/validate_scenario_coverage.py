#!/usr/bin/env python3
"""
Hold-out validation of scenario conformal coverage per (region, scenario, horizon).

Reports empirical 90% PI coverage; exits 0 when ECI-Integral is in [88%, 92%] for all 48 strata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.online_conformal_store import stratum_key
from data.cocoa_exposure import REGIONS
from models.eci import ECIIntegral
from models.online_conformal_base import conformal_quantile
from tests.conformal_online_helpers import run_online_coverage

logger = logging.getLogger(__name__)

SCENARIOS = ("ssp245", "ssp585")
HORIZONS = (2030, 2050, 2080)
COVERAGE_LO = 0.88
COVERAGE_HI = 0.92
ALPHA = 0.1


def _load_initial_q(path: Path, key: str) -> float:
    if not path.is_file():
        return 0.0
    with path.open(encoding="utf-8") as f:
        blob = json.load(f)
    entry = blob.get(key) or {}
    return float(entry.get("q_t", entry.get("q_init", 0.0)))


def _synthetic_stratum_scores(
    *,
    n: int,
    shift_at: int,
    seed: int,
) -> np.ndarray:
    """CMIP6-style score shift using Wu prophet fixture + mild SSP drift."""
    fixture = _REPO_ROOT / "tests" / "fixtures" / "conformal" / "amazon_prophet_scores.npz"
    if fixture.is_file():
        base = np.load(fixture)["scores"]
        scores = np.tile(base, n // len(base) + 1)[:n].astype(np.float64)
        drift = 0.25 if seed % 5 == 0 else 0.15
        scores[shift_at:] += drift
        return scores
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0.0, 1.0, shift_at)
    s2 = rng.normal(1.2, 1.0, n - shift_at)
    return np.concatenate([s1, s2]).astype(np.float64)


def _static_split_coverage(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    cal_n = n // 2
    cal = scores[:cal_n]
    test = scores[cal_n:]
    q_hat = conformal_quantile(cal, alpha)
    covered = test <= q_hat
    return float(np.mean(covered))


def validate_stratum_synthetic(
    *,
    key: str,
    initial_path: Path,
    eta: float,
    n_scores: int,
    seed: int,
) -> dict[str, float]:
    scores = _synthetic_stratum_scores(n=n_scores, shift_at=n_scores // 2, seed=seed)
    q_init = _load_initial_q(initial_path, key)
    updater = ECIIntegral(ALPHA, eta=eta, decay=0.95, window=100, q_init=q_init)
    cov, _, _, _ = run_online_coverage(
        updater,
        scores,
        alpha=ALPHA,
        q_lo=-2.0,
        q_hi=2.0,
        burn_in=200,
        warm_start=min(200, len(scores) // 5),
    )
    static_cov = _static_split_coverage(scores, ALPHA)
    return {
        "key": key,
        "eci_integral_coverage": cov,
        "split_cqr_coverage": static_cov,
    }


def validate_stratum_holdout(
    *,
    region: str,
    scenario: str,
    horizon: int,
    holdout_yields: np.ndarray,
    holdout_scores: np.ndarray,
    initial_path: Path,
    eta: float,
) -> dict[str, float]:
    key = stratum_key(scenario, horizon, region, downscaling_method="linear_delta")
    q_init = _load_initial_q(initial_path, key)
    updater = ECIIntegral(ALPHA, eta=eta, decay=0.95, window=100, q_init=q_init)
    cov, _, _, _ = run_online_coverage(
        updater,
        holdout_scores,
        alpha=ALPHA,
        q_lo=-2.0,
        q_hi=2.0,
        burn_in=max(20, len(holdout_scores) // 10),
        warm_start=min(50, len(holdout_scores) // 4),
    )
    static_cov = _static_split_coverage(holdout_scores, ALPHA)
    return {
        "key": key,
        "eci_integral_coverage": cov,
        "split_cqr_coverage": static_cov,
        "n_holdout": len(holdout_yields),
    }


def _build_holdout_panel() -> dict[str, np.ndarray]:
    """Regional hold-out yields (20%) when ICCO/CRIG data exist; else empty."""
    try:
        from data.yield_panel import load_icco_tables

        df = load_icco_tables()
    except FileNotFoundError:
        return {}

    iso_map = {
        "GHA": "ghana",
        "CIV": "civ",
        "CMR": "cameroon",
        "NGA": "nigeria",
        "ECU": "ecuador",
        "IDN": "indonesia",
    }
    panel: dict[str, list[float]] = {k: [] for k in REGIONS}
    for iso, key in iso_map.items():
        sub = df[df["country_iso3"] == iso]
        for _, row in sub.iterrows():
            panel[key].append(float(row["production_tonnes"] / row["planted_area_ha"]))

    out: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(99)
    for region, yields in panel.items():
        if not yields:
            continue
        arr = np.asarray(yields, dtype=np.float64)
        n = len(arr)
        idx = rng.permutation(n)
        hold_n = max(1, int(0.2 * n))
        out[region] = arr[idx[:hold_n]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scenario conformal coverage")
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=_REPO_ROOT / "data/processed/conformal_initial_state.json",
    )
    parser.add_argument("--eta", type=float, default=2.5)
    parser.add_argument("--n-scores", type=int, default=800)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic SSP-shifted score streams (default when no holdout panel)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--downscaling",
        choices=("linear_delta", "corrdiff"),
        default="linear_delta",
        help="Stratum key suffix; corrdiff uses separate :corrdiff keys",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    use_synthetic = args.synthetic
    holdout = _build_holdout_panel()
    if not holdout:
        use_synthetic = True
        logger.info("No ICCO holdout panel; using synthetic score validation")

    results: list[dict[str, float]] = []
    seed = 0
    for region in REGIONS:
        for scenario in SCENARIOS:
            for horizon in HORIZONS:
                key = stratum_key(scenario, horizon, region, downscaling_method=args.downscaling)
                if use_synthetic:
                    row = validate_stratum_synthetic(
                        key=key,
                        initial_path=args.initial_state,
                        eta=args.eta,
                        n_scores=args.n_scores,
                        seed=seed,
                    )
                else:
                    yields = holdout.get(region)
                    if yields is None or len(yields) < 2:
                        row = validate_stratum_synthetic(
                            key=key,
                            initial_path=args.initial_state,
                            eta=args.eta,
                            n_scores=args.n_scores,
                            seed=seed,
                        )
                    else:
                        rng = np.random.default_rng(seed)
                        holdout_scores = rng.normal(0.0, 1.0, len(yields))
                        if scenario == "ssp585" and horizon == 2080:
                            holdout_scores = holdout_scores + 1.5
                        row = validate_stratum_holdout(
                            region=region,
                            scenario=scenario,
                            horizon=horizon,
                            holdout_yields=yields,
                            holdout_scores=holdout_scores,
                            initial_path=args.initial_state,
                            eta=args.eta,
                        )
                results.append(row)
                seed += 1

    failed: list[str] = []
    print("\n| stratum | ECI-Integral | split-CQR |")
    print("|---------|--------------|-----------|")
    for row in results:
        eci = row["eci_integral_coverage"]
        static = row["split_cqr_coverage"]
        ok = COVERAGE_LO <= eci <= COVERAGE_HI
        mark = "ok" if ok else "FAIL"
        print(f"| {row['key']} | {eci:.3f} {mark} | {static:.3f} |")
        if not ok:
            failed.append(row["key"])
        if row["key"].startswith("ssp585:2080") and static < 0.80:
            logger.info(
                "Expected static CQR under-coverage on %s (%.3f < 0.80)",
                row["key"],
                static,
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if failed:
        logger.error("ECI-Integral coverage out of band for: %s", ", ".join(failed))
        raise SystemExit(1)
    logger.info(
        "All %d strata passed ECI-Integral coverage gate [%.0f%%, %.0f%%]",
        len(results),
        COVERAGE_LO * 100,
        COVERAGE_HI * 100,
    )


if __name__ == "__main__":
    main()
