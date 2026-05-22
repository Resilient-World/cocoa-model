"""Validate parallel-trends F-test p-value from causal evaluation JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)
MIN_PRETREND_PVALUE = 0.10


def validate_parallel_trends_report(
    report: dict,
    *,
    min_pvalue: float = MIN_PRETREND_PVALUE,
) -> None:
    pvalue = float(report.get("pretrend_pvalue", 0.0))
    if pvalue < min_pvalue:
        raise SystemExit(
            f"Parallel trends gate failed: pretrend_pvalue={pvalue:.4f} < {min_pvalue:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if parallel-trends F-test p-value < threshold",
    )
    parser.add_argument("report", type=Path, help="causal_eval.json from run_evaluation")
    parser.add_argument("--min-pvalue", type=float, default=MIN_PRETREND_PVALUE)
    args = parser.parse_args(argv)

    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)

    validate_parallel_trends_report(report, min_pvalue=args.min_pvalue)
    log.info(f"OK: pretrend_pvalue={float(report['pretrend_pvalue']):.4f} >= {args.min_pvalue:.2f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code:
            raise
        sys.exit(0)
