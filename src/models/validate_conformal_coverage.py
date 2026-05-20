"""Validate empirical conformal coverage stored in ``models/conformal.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COVERAGE_TOLERANCE = 0.02


def validate_conformal_coverage(
    payload: dict,
    *,
    tolerance: float = COVERAGE_TOLERANCE,
) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if empirical conformal coverage is too low")
    parser.add_argument("conformal_json", type=Path, help="models/conformal.json")
    parser.add_argument("--tolerance", type=float, default=COVERAGE_TOLERANCE)
    args = parser.parse_args(argv)

    with args.conformal_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    validate_conformal_coverage(payload, tolerance=args.tolerance)
    v = payload["validation"]
    print(
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
