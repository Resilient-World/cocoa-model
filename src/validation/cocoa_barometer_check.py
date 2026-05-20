"""
Cocoa Barometer 2024 regional yield-trend consistency check.

Cross-references model-predicted year-on-year yield changes against reported
anomaly directions in the Barometer (positive / negative / neutral).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from validation._report import ValidationResult, write_report
from validation.icco_yield_backtest import (
    DEFAULT_ICCO_CSV,
    load_icco_table,
    predict_national_production,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAROMETER_JSON = _REPO_ROOT / "data" / "external" / "cocoa_barometer_2024_anomalies.json"
DEFAULT_REPORT = _REPO_ROOT / "reports" / "validation" / "cocoa_barometer_check.md"

AGREEMENT_GATE = 0.60  # 60% directional match


def load_barometer_anomalies(path: Path | None = None) -> dict[str, dict[str, str]]:
    json_path = path or DEFAULT_BAROMETER_JSON
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["regions"]


def _sign_from_delta(delta: float, tol: float = 0.02) -> str:
    if delta > tol:
        return "positive"
    if delta < -tol:
        return "negative"
    return "neutral"


def predicted_yoy_signs(icco_df: pd.DataFrame, yield_factors: dict[str, float] | None) -> pd.DataFrame:
    """Derive YoY production change sign from aggregated model production forecasts."""
    preds = predict_national_production(icco_df, country_yield_factors=yield_factors)
    rows: list[dict] = []
    for iso, group in preds.groupby("country_iso3"):
        g = group.sort_values("year")
        prod = g["predicted_production_tonnes"].astype(float)
        for i in range(1, len(g)):
            delta = float(prod.iloc[i] - prod.iloc[i - 1]) / max(float(prod.iloc[i - 1]), 1.0)
            rows.append(
                {
                    "country_iso3": iso,
                    "year": int(g["year"].iloc[i]),
                    "predicted_sign": _sign_from_delta(delta),
                }
            )
    return pd.DataFrame(rows)


def run_barometer_check(
    barometer_path: Path | None = None,
    icco_path: Path | None = None,
    *,
    yield_factors: dict[str, float] | None = None,
) -> ValidationResult:
    barometer = load_barometer_anomalies(barometer_path)
    icco = load_icco_table(icco_path)
    pred = predicted_yoy_signs(icco, yield_factors)
    matches = 0
    total = 0
    mismatches: list[str] = []

    for _, row in pred.iterrows():
        iso = str(row["country_iso3"])
        year = str(int(row["year"]))
        reported = barometer.get(iso, {}).get(year)
        if reported is None:
            continue
        total += 1
        if row["predicted_sign"] == reported:
            matches += 1
        else:
            mismatches.append(f"{iso}-{year}: pred={row['predicted_sign']} vs barometer={reported}")

    agreement = matches / total if total else 0.0
    metrics = {
        "directional_agreement": agreement,
        "n_comparisons": total,
        "n_mismatches": len(mismatches),
    }
    passed = agreement >= AGREEMENT_GATE
    return ValidationResult(
        name="Cocoa Barometer 2024 trend check",
        passed=passed,
        metrics=metrics,
        gate_description=f"Directional agreement ≥ {AGREEMENT_GATE:.0%} with Barometer anomalies",
        notes=mismatches[:10] if mismatches else ["All compared years match Barometer signs"],
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cocoa Barometer trend validation")
    parser.add_argument("--barometer-json", type=Path, default=DEFAULT_BAROMETER_JSON)
    parser.add_argument("--icco-csv", type=Path, default=DEFAULT_ICCO_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    result = run_barometer_check(args.barometer_json, args.icco_csv)
    write_report(result, args.report)
    print(
        f"Barometer check: {'PASS' if result.passed else 'FAIL'} "
        f"(agreement={result.metrics['directional_agreement']:.1%})"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
