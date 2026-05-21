#!/usr/bin/env python3
"""
Validate WCTM drift detection: synthetic null, covariate, concept, and joint shifts.

Writes ``reports/monitoring/wctm_validation_<date>.md``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from monitoring.conformal_cusum import ConformalCUSUM
from monitoring.wctm import WeightedConformalTestMartingale, covariate_nonconformity

logger = logging.getLogger(__name__)

REPORT_DIR = _REPO_ROOT / "reports" / "monitoring"
PAPER_SHIFT_AT = 500
DELAY_TOLERANCE = 0.20
ALPHA_FPR = 0.01
N_SEEDS = 20
T_DEFAULT = 1200


def _run_null_stream(
    *,
    t: int,
    seed: int,
    alpha_fpr: float,
) -> tuple[bool, float]:
    rng = np.random.default_rng(seed)
    wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    for _ in range(t):
        score = float(rng.uniform(0.0, 2.0))
        wctm.update(score, weight=1.0)
    return wctm.detect() is not None, wctm.log_martingale


def _run_concept_shift(
    *,
    t: int,
    shift_at: int,
    seed: int,
    alpha_fpr: float,
) -> int | None:
    rng = np.random.default_rng(seed)
    wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    x_wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    ema: list[float] | None = None
    delay: int | None = None
    for step in range(t):
        if step < shift_at:
            score = float(rng.normal(0.0, 0.5))
            feat = [float(rng.normal(0, 0.1)) for _ in range(16)]
        else:
            score = float(rng.normal(3.5, 0.8))
            feat = [float(rng.normal(0, 0.1)) for _ in range(16)]
        x_score, ema = covariate_nonconformity(feat, ema)
        wctm.update(score, weight=1.0)
        x_wctm.update(x_score, weight=1.0)
        x_alarm = x_wctm.log_martingale > x_wctm.log_threshold
        wctm.set_x_alarm_active(x_alarm)
        if delay is None and step >= shift_at and wctm.detect() is not None:
            if wctm.diagnose() == "concept_shift":
                delay = step - shift_at
    return delay


def _run_covariate_shift(
    *,
    t: int,
    shift_at: int,
    seed: int,
    alpha_fpr: float,
) -> int | None:
    rng = np.random.default_rng(seed)
    wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    x_wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    ema: list[float] | None = None
    delay: int | None = None
    for step in range(t):
        if step < shift_at:
            score = float(rng.normal(0.0, 0.5))
            feat = [float(rng.normal(0, 0.1)) for _ in range(16)]
        else:
            score = float(rng.normal(0.5, 0.5))
            feat = [float(rng.normal(4.0, 1.0)) for _ in range(16)]
        x_score, ema = covariate_nonconformity(feat, ema)
        wctm.update(score, weight=1.0)
        x_wctm.update(x_score, weight=1.0)
        x_alarm = x_wctm.log_martingale > x_wctm.log_threshold
        wctm.set_x_alarm_active(x_alarm)
        if delay is None and step >= shift_at and wctm.detect() is not None:
            if wctm.diagnose() == "covariate_shift":
                delay = step - shift_at
    return delay


def _run_joint_shift(
    *,
    t: int,
    shift_at: int,
    seed: int,
    alpha_fpr: float,
) -> bool:
    rng = np.random.default_rng(seed)
    wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    x_wctm = WeightedConformalTestMartingale(alpha_fpr=alpha_fpr)
    ema: list[float] | None = None
    fired = False
    for step in range(t):
        if step < shift_at:
            score = float(rng.normal(0.0, 0.5))
            feat = [float(rng.normal(0, 0.1)) for _ in range(16)]
        else:
            score = float(rng.normal(3.0, 0.8))
            feat = [float(rng.normal(4.0, 1.0)) for _ in range(16)]
        x_score, ema = covariate_nonconformity(feat, ema)
        wctm.update(score, weight=1.0)
        x_wctm.update(x_score, weight=1.0)
        wctm.set_x_alarm_active(x_wctm.log_martingale > x_wctm.log_threshold)
        if step >= shift_at and wctm.detect() is not None:
            fired = True
    return fired


def _cusum_agreement(t: int, shift_at: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    wctm = WeightedConformalTestMartingale(alpha_fpr=ALPHA_FPR)
    cusum = ConformalCUSUM(h=4.0, k=0.0)
    agree = 0
    total = 0
    for step in range(t):
        score = float(rng.normal(0.0, 0.5)) if step < shift_at else float(rng.normal(2.5, 0.6))
        wctm.update(score)
        cusum.update(score)
        if step >= shift_at + 50:
            total += 1
            agree += int(wctm.detect() is not None) == int(cusum.detect())
    return float(agree / total) if total else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate WCTM drift detection")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--t", type=int, default=T_DEFAULT)
    parser.add_argument("--shift-at", type=int, default=PAPER_SHIFT_AT)
    parser.add_argument("--alpha-fpr", type=float, default=ALPHA_FPR)
    parser.add_argument("--quick", action="store_true", help="Fewer seeds and shorter stream")
    parser.add_argument("--plot", action="store_true", help="Write PNG figures (requires matplotlib)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_seeds = 5 if args.quick else args.n_seeds
    t = 600 if args.quick else args.t
    shift_at = min(args.shift_at, t // 2)

    null_alarms = []
    concept_delays: list[int] = []
    covariate_delays: list[int] = []
    joint_hits = []
    cusum_rates: list[float] = []

    for seed in range(n_seeds):
        alarm, _ = _run_null_stream(t=t, seed=seed, alpha_fpr=args.alpha_fpr)
        null_alarms.append(alarm)
        d = _run_concept_shift(t=t, shift_at=shift_at, seed=seed + 100, alpha_fpr=args.alpha_fpr)
        if d is not None:
            concept_delays.append(d)
        d2 = _run_covariate_shift(t=t, shift_at=shift_at, seed=seed + 200, alpha_fpr=args.alpha_fpr)
        if d2 is not None:
            covariate_delays.append(d2)
        joint_hits.append(_run_joint_shift(t=t, shift_at=shift_at, seed=seed + 300, alpha_fpr=args.alpha_fpr))
        cusum_rates.append(_cusum_agreement(t, shift_at, seed + 400))

    far = float(np.mean(null_alarms))
    med_concept = float(np.median(concept_delays)) if concept_delays else float("nan")
    med_covariate = float(np.median(covariate_delays)) if covariate_delays else float("nan")
    joint_rate = float(np.mean(joint_hits))
    cusum_agree = float(np.mean(cusum_rates))

    lo_delay = int(PAPER_SHIFT_AT * (1.0 - DELAY_TOLERANCE))
    hi_delay = int(PAPER_SHIFT_AT * (1.0 + DELAY_TOLERANCE))
    concept_ok = concept_delays and lo_delay <= med_concept <= hi_delay
    covariate_ok = covariate_delays and med_covariate <= hi_delay
    far_ok = far <= args.alpha_fpr * 2.5

    today = date.today().isoformat()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"wctm_validation_{today}.md"
    lines = [
        f"# WCTM validation ({today})",
        "",
        "## Configuration",
        "",
        f"- Streams per scenario: {n_seeds} seeds × {t} steps",
        f"- Changepoint at t={shift_at} (paper baseline ~{PAPER_SHIFT_AT}, ±{int(DELAY_TOLERANCE*100)}% delay window)",
        f"- `alpha_fpr={args.alpha_fpr}` (alarm when log-martingale > {math.log(1/args.alpha_fpr):.3f})",
        "",
        "## Results",
        "",
        "| Metric | Value | Gate |",
        "|--------|-------|------|",
        f"| Null false-alarm rate | {far:.3f} | ≤ {args.alpha_fpr * 2.5:.3f} |",
        f"| Concept-shift median delay | {med_concept:.0f} | [{lo_delay}, {hi_delay}] |",
        f"| Covariate-shift median delay | {med_covariate:.0f} | ≤ {hi_delay} |",
        f"| Joint-shift detection rate | {joint_rate:.2f} | ≥ 0.8 |",
        f"| CUSUM agreement (post-shift) | {cusum_agree:.2f} | informational |",
        "",
        "## Interpretation",
        "",
        "- **Null:** i.i.d. scores should rarely cross the Ville threshold.",
        "- **Concept:** label scores jump at `shift_at`; X-CTM stays quiet → `concept_shift`.",
        "- **Covariate:** feature distance jumps; both WCTM and X-CTM alarm → `covariate_shift`.",
        "- **CUSUM:** parallel IID change-point sanity check (Vovk et al., PMLR 266).",
        "",
        f"**Overall:** {'PASS' if far_ok and concept_ok and joint_rate >= 0.8 else 'REVIEW'}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skip --plot")
        else:
            fig_dir = REPORT_DIR / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(0)
            wctm = WeightedConformalTestMartingale(alpha_fpr=args.alpha_fpr)
            log_path = []
            for step in range(t):
                score = float(rng.normal(0.0, 0.5)) if step < shift_at else float(rng.normal(3.0, 0.7))
                log_path.append(wctm.update(score))
            plt.figure(figsize=(8, 3))
            plt.plot(log_path)
            plt.axvline(shift_at, color="red", linestyle="--", label="shift")
            plt.axhline(math.log(1 / args.alpha_fpr), color="orange", linestyle=":", label="threshold")
            plt.title("WCTM log-martingale (concept shift example)")
            plt.legend()
            plt.tight_layout()
            fig_path = fig_dir / f"wctm_log_martingale_{today}.png"
            plt.savefig(fig_path, dpi=120)
            plt.close()
            logger.info("Wrote %s", fig_path)

    return 0 if far_ok and concept_ok and joint_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
