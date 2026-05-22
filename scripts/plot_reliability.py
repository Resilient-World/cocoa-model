#!/usr/bin/env python3
"""Reliability curve, PIT histogram, and CRPSS bar chart for calibration reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot calibration diagnostics")
    parser.add_argument("--model", default="cqr_yield")
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="calibration JSON (default: reports/validation/calibration_latest.json)",
    )
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "reports" / "validation")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plot_reliability", file=sys.stderr)
        return 1

    scores_path = args.scores or (args.out_dir / "calibration_latest.json")
    if not scores_path.is_file():
        print(f"Scores file not found: {scores_path}", file=sys.stderr)
        return 1
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    day = args.date or data.get("date") or date.today().isoformat()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"reliability_{args.model}_{day}.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    nom = data.get("reliability_nominal") or []
    emp = data.get("reliability_empirical") or []
    ax0 = axes[0]
    if nom and emp:
        ax0.plot(nom, emp, "o-", label="Model")
        ax0.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax0.set_xlabel("Nominal quantile")
    ax0.set_ylabel("Empirical frequency")
    ax0.set_title(f"Reliability (ECE={data.get('ece', float('nan')):.3f})")
    ax0.legend(loc="lower right", fontsize=8)

    pit = data.get("pit") or []
    ax1 = axes[1]
    if pit:
        ax1.hist(pit, bins=10, range=(0, 1), density=True, alpha=0.75)
        ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax1.set_xlabel("PIT")
    ax1.set_title(
        f"PIT p={data.get('pit_chi2_p', float('nan')):.3f} "
        f"({data.get('pit_shape', '?')})"
    )

    ax2 = axes[2]
    labels = ["clim", "pers", "fdp"]
    keys = ["crpss_climatology", "crpss_persistence", "crpss_fdp_mean"]
    vals = [float(data.get(k, float("nan"))) for k in keys]
    colors = ["#2a6f97" if np.isfinite(v) and v > 0 else "#e76f51" for v in vals]
    ax2.bar(labels, vals, color=colors)
    ax2.axhline(0.0, color="k", linewidth=0.8)
    ax2.set_ylabel("CRPSS")
    ax2.set_title(f"CRPSS (CRPS={data.get('crps', float('nan')):.3f})")

    fig.suptitle(f"Calibration — {args.model} ({day})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
