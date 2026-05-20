"""Validate max |SMD| from a causal evaluation JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_SMD_THRESHOLD = 0.10


def validate_smd_report(report: dict, *, threshold: float = MAX_SMD_THRESHOLD) -> None:
    max_smd = float(report.get("max_smd", report.get("max_smd_matched", 1.0)))
    if max_smd >= threshold:
        raise SystemExit(
            f"SMD balance gate failed: max_smd={max_smd:.4f} >= {threshold:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if max |SMD| >= threshold")
    parser.add_argument("report", type=Path, help="causal_eval.json from run_evaluation")
    parser.add_argument("--threshold", type=float, default=MAX_SMD_THRESHOLD)
    args = parser.parse_args(argv)

    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)

    validate_smd_report(report, threshold=args.threshold)
    print(f"OK: max_smd={float(report.get('max_smd', 0)):.4f} < {args.threshold:.2f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code:
            raise
        sys.exit(0)
