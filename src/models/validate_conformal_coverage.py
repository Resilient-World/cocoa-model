"""Validate conformal coverage and benchmark online calibration methods."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models.aci import AdaptiveConformalInference
from models.conformal_pid import ConformalPID
from models.cqr import ConformalCalibrator
from models.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from models.online_conformal_base import (
    empirical_coverage,
    interval_from_q,
    mean_interval_width,
    pit_values,
    rolling_coverage,
)

COVERAGE_TOLERANCE = 0.02

MethodRunner = Callable[[np.ndarray, float], dict[str, Any]]


def validate_conformal_coverage(
    payload: dict,
    *,
    tolerance: float = COVERAGE_TOLERANCE,
) -> None:
    """Gate legacy ``models/conformal.json`` empirical coverage."""
    validation = payload.get("validation") or {}
    empirical = validation.get("empirical_coverage")
    nominal = validation.get(
        "nominal_coverage",
        payload.get("coverage_target", 1.0 - float(payload.get("alpha", 0.1))),
    )
    if empirical is None:
        raise SystemExit(
            "conformal.json missing validation.empirical_coverage — re-run calibrate with --record-validation"
        )
    empirical_f = float(empirical)
    nominal_f = float(nominal)
    floor = nominal_f - tolerance
    if empirical_f < floor:
        raise SystemExit(
            f"Conformal coverage gate failed: empirical={empirical_f:.4f} < "
            f"nominal−{tolerance:.2f} ({floor:.4f})"
        )


def distribution_shift_simulation(
    T: int = 1000,
    shift_at: int = 500,
    *,
    alpha: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Scores from N(0,1) then N(2,1); compare online methods vs static split-CQR."""
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0.0, 1.0, shift_at)
    s2 = rng.normal(2.0, 1.0, T - shift_at)
    scores = np.concatenate([s1, s2]).astype(np.float64)
    return run_online_calibration_comparison(
        scores,
        np.zeros(T),
        alpha=alpha,
        q_lo=-1.0,
        q_hi=1.0,
        include_static_split=True,
        cal_fraction=0.5,
    )


def _stream_method(
    scores: np.ndarray,
    alpha: float,
    updater: Any,
    *,
    q_lo: float,
    q_hi: float,
    y: np.ndarray,
    burn_in: int,
) -> dict[str, Any]:
    n = len(scores)
    lowers = np.empty(n)
    uppers = np.empty(n)
    qs = np.empty(n)
    covered = np.empty(n, dtype=bool)
    updater.reset(q_init=0.0)
    for t in range(n):
        q = updater.current_threshold
        qs[t] = q
        covered[t] = scores[t] <= q
        lo, _, hi = interval_from_q(q, q_lo, q_hi)
        lowers[t] = lo
        uppers[t] = hi
        updater.update(float(scores[t]))
    sl = slice(burn_in, n)
    cov = empirical_coverage(y[sl], lowers[sl], uppers[sl])
    width = mean_interval_width(lowers[sl], uppers[sl])
    roll = rolling_coverage(covered.astype(float), window=50)
    pit = pit_values(scores[sl], qs[sl])
    return {
        "coverage": cov,
        "mean_width": width,
        "thresholds": qs,
        "lowers": lowers,
        "uppers": uppers,
        "rolling_coverage": roll,
        "pit": pit,
        "final_q": float(qs[-1]),
    }


def run_online_calibration_comparison(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.1,
    q_lo: float = -1.0,
    q_hi: float = 1.0,
    burn_in: int = 100,
    include_static_split: bool = False,
    cal_fraction: float = 0.5,
) -> dict[str, Any]:
    """Benchmark ACI, PID, ECI variants on a score stream."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) != len(scores):
        y = np.zeros_like(scores)

    methods: dict[str, MethodRunner] = {
        "aci": lambda s, a: _stream_method(
            s, a, AdaptiveConformalInference(a, eta=0.02), q_lo=q_lo, q_hi=q_hi, y=y, burn_in=burn_in
        ),
        "conformal_pid": lambda s, a: _stream_method(
            s, a, ConformalPID(a, eta=0.02, window=100), q_lo=q_lo, q_hi=q_hi, y=y, burn_in=burn_in
        ),
        "eci": lambda s, a: _stream_method(
            s,
            a,
            ErrorQuantifiedConformalInference(a, eta=0.02, window=100),
            q_lo=q_lo,
            q_hi=q_hi,
            y=y,
            burn_in=burn_in,
        ),
        "eci_cutoff": lambda s, a: _stream_method(
            s, a, ECICutoff(a, eta=0.02, window=100), q_lo=q_lo, q_hi=q_hi, y=y, burn_in=burn_in
        ),
        "eci_integral": lambda s, a: _stream_method(
            s, a, ECIIntegral(a, eta=0.02, window=100), q_lo=q_lo, q_hi=q_hi, y=y, burn_in=burn_in
        ),
    }

    results: dict[str, Any] = {"alpha": alpha, "n": len(scores), "burn_in": burn_in, "methods": {}}
    for name, runner in methods.items():
        results["methods"][name] = runner(scores, alpha)

    if include_static_split:
        n_cal = int(len(scores) * cal_fraction)
        Q = ConformalCalibrator._conformal_quantile(scores[:n_cal], alpha)
        test = scores[n_cal:]
        results["methods"]["split_cqr_static"] = {
            "coverage": float(np.mean(test <= Q)),
            "mean_width": float((q_hi + Q) - (q_lo - Q)),
            "Q_hat": Q,
        }
    return results


def write_online_calibration_report(
    out_dir: Path,
    results: dict[str, Any],
    *,
    report_date: str | None = None,
) -> Path:
    """Write ``reports/conformal/online_calibration_<date>.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report_date or date.today().isoformat()
    md_path = out_dir / f"online_calibration_{day}.md"

    lines = [
        f"# Online conformal calibration benchmark ({day})",
        "",
        f"- α = {results.get('alpha', 0.1):.2f} (nominal coverage {(1 - results.get('alpha', 0.1)):.0%})",
        f"- n = {results.get('n', 0)} timesteps, burn-in = {results.get('burn_in', 0)}",
        "",
        "## Method summary",
        "",
        "| Method | Empirical coverage | Mean width | Final q |",
        "|--------|-------------------|------------|---------|",
    ]
    for name, stats in results.get("methods", {}).items():
        if not isinstance(stats, dict) or "coverage" not in stats:
            continue
        lines.append(
            f"| {name} | {stats['coverage']:.3f} | {stats.get('mean_width', float('nan')):.4f} | "
            f"{stats.get('final_q', stats.get('Q_hat', '')):.4f} |"
            if "final_q" in stats
            else f"| {name} | {stats['coverage']:.3f} | {stats.get('mean_width', float('nan')):.4f} | — |"
        )

    lines.extend(["", "## PIT uniformity (binned)", ""])
    for name, stats in results.get("methods", {}).items():
        if not isinstance(stats, dict) or "pit" not in stats:
            continue
        pit = np.asarray(stats["pit"])
        hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
        lines.append(f"### {name}")
        lines.append(f"- histogram: {hist.tolist()}")
        lines.append("")

    _try_write_pit_plots(out_dir, results, day)

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _try_write_pit_plots(out_dir: Path, results: dict[str, Any], day: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = results.get("methods", {})
    for name, stats in methods.items():
        if not isinstance(stats, dict) or "pit" not in stats:
            continue
        pit = np.asarray(stats["pit"])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(pit, bins=10, range=(0, 1), density=True, alpha=0.7)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"PIT — {name}")
        fig.savefig(out_dir / f"pit_{name}_{day}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def run_benchmark_online(
    out_dir: Path,
    *,
    T: int = 1000,
    shift_at: int = 500,
    alpha: float = 0.1,
) -> Path:
    results = distribution_shift_simulation(T=T, shift_at=shift_at, alpha=alpha)
    return write_online_calibration_report(out_dir, results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conformal coverage validation and online benchmarks")
    parser.add_argument("conformal_json", type=Path, nargs="?", help="models/conformal.json (legacy gate)")
    parser.add_argument("--tolerance", type=float, default=COVERAGE_TOLERANCE)
    parser.add_argument(
        "--benchmark-online",
        action="store_true",
        help="Run shift simulation and write reports/conformal/online_calibration_<date>.md",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/conformal"))
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--shift-at", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args(argv)

    if args.benchmark_online:
        path = run_benchmark_online(args.out, T=args.T, shift_at=args.shift_at, alpha=args.alpha)
        print(f"Wrote {path}")
        return 0

    if args.conformal_json is None:
        parser.error("conformal_json required unless --benchmark-online is set")

    with args.conformal_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    validate_conformal_coverage(payload, tolerance=args.tolerance)
    v = payload["validation"]
    print(
        f"OK: empirical_coverage={float(v['empirical_coverage']):.4f} "
        f">= {float(v['nominal_coverage']) - args.tolerance:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code:
            raise
        sys.exit(0)
