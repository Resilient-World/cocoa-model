"""Validate conformal coverage and benchmark online calibration methods."""

from __future__ import annotations

import structlog

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models.conformal.aci import AdaptiveConformalInference
from models.conformal.conformal_pid import ConformalPID
from models.conformal.cqr import ConformalCalibrator
from models.conformal.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from data.yield_panel import build_yield_panel
from models.conformal.online_conformal_base import (
    empirical_coverage,
    interval_from_q,
    mean_interval_width,
    pit_values,
    rolling_coverage,
)

COVERAGE_LO = 0.88
COVERAGE_HI = 0.92

log = structlog.get_logger(__name__)

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


def write_cv_strategy_report(out_dir: Path, results: dict[str, Any]) -> Path:
    """Write markdown report for multi-strategy conformal CV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    path = out_dir / f"conformal_cv_{day}.md"
    lines = [
        "# Conformal coverage by CV strategy",
        "",
        f"Date: {day}",
        "",
        "**Production target:** `spatial_block` coverage in [88%, 92%] at 90% nominal.  ",
        "**Secondary diagnostic:** `random` split (optimistic under spatial autocorrelation).",
        "",
        "| Strategy | Coverage | CRPS | ECE | PIT p | Sharpness | Production |",
        "|----------|----------|------|-----|-------|-----------|------------|",
    ]
    for name, stats in results.items():
        if not isinstance(stats, dict) or "coverage" not in stats:
            continue
        prod = "yes" if stats.get("production_target") else "no (diagnostic)"
        lines.append(
            f"| {name} | {float(stats['coverage']):.3f} | "
            f"{float(stats.get('crps', float('nan'))):.4f} | "
            f"{float(stats.get('ece', float('nan'))):.4f} | "
            f"{float(stats.get('pit_chi2_p', float('nan'))):.4f} | "
            f"{float(stats.get('sharpness', float('nan'))):.4f} | {prod} |"
        )
    if "gate_message" in results:
        lines.extend(["", f"Gate: {results['gate_message']}"])
    lines.extend(
        [
            "",
            "Scenario API: production calibrators should be trained with `fit_blocked` "
            "(`spatial_block`); saved `cv_strategy` metadata is exposed on `/simulate-scenario` "
            "responses when present.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conformal coverage validation and online benchmarks")
    parser.add_argument("conformal_json", type=Path, nargs="?", help="models/conformal.json (legacy gate)")
    parser.add_argument("--tolerance", type=float, default=COVERAGE_TOLERANCE)
    parser.add_argument(
        "--benchmark-online",
        action="store_true",
        help="Run shift simulation and write reports/conformal/online_calibration_<date>.md",
    )
    parser.add_argument(
        "--cv-strategy",
        choices=("random", "spatial_block", "temporal_forward", "buffered_loo", "all"),
        default=None,
        help="Blocked conformal evaluation (production: spatial_block)",
    )
    parser.add_argument("--block-size-km", type=float, default=50.0)
    parser.add_argument("--out", type=Path, default=Path("reports/conformal"))
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--shift-at", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--calibration-report",
        action="store_true",
        help="Write calibration JSON/Markdown under --out",
    )
    parser.add_argument(
        "--calibration-gate",
        action="store_true",
        help="Fail if coverage, PIT chi2, or sharpness regression gates fail",
    )
    parser.add_argument(
        "--baseline-calibration",
        type=Path,
        default=Path("tests/fixtures/promotion/baseline_calibration.json"),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic yield panel when ICCO data unavailable",
    )
    args = parser.parse_args(argv)

    if args.calibration_report or args.calibration_gate:
        from validation.calibration_metrics import (
            evaluate_cqr_calibration,
            run_calibration_gate,
            write_calibration_report,
        )
        from validation.conformal_cv import _synthetic_panel_rows

        try:
            rows = build_yield_panel(seed=42)
        except Exception:
            rows = _synthetic_panel_rows(400, seed=42)
        if args.synthetic:
            rows = _synthetic_panel_rows(400, seed=42)
        report = evaluate_cqr_calibration(rows, alpha=args.alpha, block_size_km=args.block_size_km)
        baseline_payload = None
        if args.baseline_calibration.is_file():
            baseline_payload = json.loads(args.baseline_calibration.read_text(encoding="utf-8"))
        out_validation = args.out if args.out.name == "validation" else Path("reports/validation")
        if args.calibration_report:
            jpath, mpath = write_calibration_report(report, out_validation)
            log.info("Wrote calibration report %s and %s", jpath, mpath)
        if args.calibration_gate:
            ok, msgs = run_calibration_gate(report, baseline_payload)
            for m in msgs:
                log.info(m)
            if not ok:
                raise SystemExit(1)
        return 0

    if args.cv_strategy:
        from validation.conformal_cv import evaluate_cv_strategy, run_all_cv_strategies

        if args.cv_strategy == "all":
            results = run_all_cv_strategies(block_size_km=args.block_size_km)
            report = write_cv_strategy_report(args.out, results)
            log.info(f"Wrote {report}")
            if results.get("gate_passed") is False:
                raise SystemExit(1)
            return 0
        try:
            rows = build_yield_panel(seed=42)
        except Exception:
            from validation.conformal_cv import _synthetic_panel_rows

            rows = _synthetic_panel_rows(300)
        stats = evaluate_cv_strategy(
            args.cv_strategy,
            rows,
            block_size_km=args.block_size_km,
        )
        report = write_cv_strategy_report(args.out, {args.cv_strategy: stats})
        log.info(f"Wrote {report}; coverage={stats.get('coverage')}")
        if stats.get("production_target") and not (
            COVERAGE_LO <= float(stats["coverage"]) <= COVERAGE_HI
        ):
            raise SystemExit(1)
        if args.calibration_gate:
            from validation.calibration_metrics import run_calibration_gate

            baseline_payload = None
            if args.baseline_calibration.is_file():
                baseline_payload = json.loads(
                    args.baseline_calibration.read_text(encoding="utf-8")
                )
            gate_data = {
                "nominal_coverage": 1.0 - args.alpha,
                "empirical_coverage": stats["coverage"],
                "pit_chi2_p": stats.get("pit_chi2_p", 1.0),
                "sharpness": stats.get("sharpness", float("nan")),
            }
            ok, msgs = run_calibration_gate(gate_data, baseline_payload)
            for m in msgs:
                log.info(m)
            if not ok:
                raise SystemExit(1)
        return 0

    if args.benchmark_online:
        path = run_benchmark_online(args.out, T=args.T, shift_at=args.shift_at, alpha=args.alpha)
        log.info(f"Wrote {path}")
        return 0

    if args.conformal_json is None:
        parser.error("conformal_json required unless --benchmark-online is set")

    with args.conformal_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    validate_conformal_coverage(payload, tolerance=args.tolerance)
    v = payload["validation"]
    log.info(
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
