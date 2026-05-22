#!/usr/bin/env python3
"""Forward-chaining temporal holdout for yield + CQR (2018–2024)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from validation.conformal_cv import _synthetic_panel_rows, evaluate_cv_strategy
from validation.temporal_cv import iter_forward_folds

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal forward-chain validation")
    parser.add_argument("--min-train-years", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "reports" / "validation")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = 200 if args.quick else 500
    rows = _synthetic_panel_rows(n, seed=42)
    years = np.array([r.year for r in rows])

    stats = evaluate_cv_strategy("temporal_forward", rows, seed=42)
    fold_count = sum(
        1
        for _ in iter_forward_folds(
            years,
            min_train_years=args.min_train_years,
            max_test_years=1,
        )
    )

    pit = np.asarray(stats.get("pit", []), dtype=np.float64)
    day = date.today().isoformat()
    out_md = args.out_dir / f"temporal_cv_{day}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Temporal forward-chain validation",
        "",
        f"Date: {day}",
        f"Forward folds (synthetic panel): {fold_count}",
        "",
        f"CRPS proxy / coverage (last fold): {float(stats.get('coverage', float('nan'))):.3f}",
        f"Mean interval width: {float(stats.get('mean_width', float('nan'))):.3f}",
        "",
        "## PIT histogram (binned)",
    ]
    if len(pit):
        hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
        lines.append(f"- counts: {hist.tolist()}")
    else:
        lines.append("- (no PIT values — run full panel with ICCO years)")
    lines.append("")
    lines.append("Chronological order preserved: no future years in training folds.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        if len(pit):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(pit, bins=10, range=(0, 1), density=True, alpha=0.7)
            ax.axhline(1.0, color="k", linestyle="--")
            ax.set_title("PIT — temporal forward")
            fig_path = args.out_dir / "figures" / f"pit_temporal_{day}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    logger.info("Wrote %s", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
