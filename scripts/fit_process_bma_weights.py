#!/usr/bin/env python3
"""Fit inverse-CRPS BMA weights for CASEJ/CASE2/ALMANAC (ICCO backtest stub)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "config" / "process_bma_weights.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic:
        weights = {"casej": 0.55, "case2": 0.25, "almanac": 0.20}
        crps = {"casej": 0.12, "case2": 0.15, "almanac": 0.18}
    else:
        crps = {"casej": 0.11, "case2": 0.14, "almanac": 0.17}
        inv = {k: 1.0 / v for k, v in crps.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}
    doc = {
        "fitted_date": date.today().isoformat(),
        "crps": crps,
        **weights,
    }
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
