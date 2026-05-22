"""Run all external validation benchmarks (DVC ``validate`` stage)."""

from __future__ import annotations

import structlog

import argparse
import logging
import sys
from pathlib import Path

from validation._report import combine_summary, write_report
from validation.cocoa_barometer_check import run_barometer_check
from validation.giews_drought_validation import run_giews_validation
from validation.icco_yield_backtest import run_icco_backtest
from validation.kalischek_benchmark import run_kalischek_benchmark

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = _REPO_ROOT / "reports" / "validation"


def run_all(
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    use_gee: bool = False,
    segmentation_ckpt: Path | None = None,
    fail_fast: bool = True,
) -> list:
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_kalischek_benchmark(use_gee=use_gee, segmentation_ckpt=segmentation_ckpt),
        run_icco_backtest(),
        run_barometer_check(),
        run_giews_validation(),
    ]
    names = [
        "kalischek_benchmark",
        "icco_yield_backtest",
        "cocoa_barometer_check",
        "giews_drought_validation",
    ]
    for name, result in zip(names, results, strict=True):
        write_report(result, reports_dir / f"{name}.md")

    summary_path = reports_dir / "summary.md"
    summary_path.write_text(combine_summary(results), encoding="utf-8")

    failed = [r for r in results if not r.passed]
    if failed:
        log.error("Validation failed: %s", ", ".join(r.name for r in failed))
        if fail_fast:
            raise SystemExit(1)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="External validation suite")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--use-gee", action="store_true")
    parser.add_argument(
        "--segmentation-ckpt",
        type=Path,
        default=_REPO_ROOT / "models" / "segmentation.ckpt",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Write all reports even when a gate fails",
    )
    args = parser.parse_args(argv)

    try:
        run_all(
            reports_dir=args.reports_dir,
            use_gee=args.use_gee,
            segmentation_ckpt=args.segmentation_ckpt,
            fail_fast=not args.no_fail_fast,
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
