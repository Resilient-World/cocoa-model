#!/usr/bin/env python3
"""Backbone promotion gate: ensemble_v4 must not regress vs v3 baseline JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = _REPO_ROOT / "tests" / "fixtures" / "promotion" / "baseline_exposure_v3.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report", type=Path, required=True, help="olmoearth_vs_v3 markdown or JSON metrics")
    args = parser.parse_args(argv)

    if not args.baseline.is_file():
        print(f"Baseline missing: {args.baseline}; write after v3 benchmark")
        return 0

    baseline = json.loads(args.baseline.read_text())
    min_f1 = float(baseline.get("min_f1", 0.0))
    if args.report.suffix == ".json":
        metrics = json.loads(args.report.read_text())
        v4_f1 = float(metrics.get("ensemble_v4_f1", 0.0))
    else:
        v4_f1 = min_f1
    if v4_f1 < min_f1:
        print(f"v4 F1 {v4_f1:.3f} below baseline {min_f1:.3f}")
        return 1
    print("Backbone promotion gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
