#!/usr/bin/env python3
"""Generate sample TCAV plots under reports/tcav/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from analysis.tcav import tcav_scores  # noqa: E402
from models.surrogate.yield_surrogate import YieldSurrogateModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "reports" / "tcav")
    args = parser.parse_args(argv)
    model = YieldSurrogateModel()
    model.eval()
    climate = torch.randn(8, 365, 11)
    static = torch.randn(8, 13)
    results = tcav_scores(model, climate=climate, static=static)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r.concept for r in results], [r.score for r in results])
    ax.set_ylabel("TCAV score")
    ax.set_ylim(0, 1)
    fig.savefig(args.out_dir / "tcav_sample.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(args.out_dir / "tcav_sample.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
