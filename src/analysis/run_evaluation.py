"""
Run causal evaluation (PSM balance + DiD + parallel trends) and write JSON report.

Output schema is consumed by :mod:`analysis.validate_smd` and
:mod:`analysis.validate_parallel_trends`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from analysis.did_comparison_harness import compare_did_methods
from analysis.did_impact import calculate_did_att, event_study
from analysis.psm_matching import propensity_score_match, standardized_mean_differences

log = structlog.get_logger(__name__)

DEFAULT_COVARIATES = (
    "farm_size_ha",
    "baseline_yield",
    "soil_quality_index",
    "historical_rainfall",
)


def _synthetic_panel(n: int, seed: int) -> pd.DataFrame:
    """Balanced panel when ``farm_panel.parquet`` is absent (CI / smoke tests)."""
    rng = np.random.default_rng(seed)
    farms = np.arange(n // 2)
    rows: list[dict[str, Any]] = []
    for farm_id in farms:
        treated = int(farm_id % 2 == 0)
        for period in (0, 1):
            rows.append(
                {
                    "farm_id": f"F{farm_id}",
                    "period": period,
                    "treated": treated,
                    "role": "treated" if treated else "control",
                    "yield_tpha": 1.5 + 0.3 * treated * period + rng.normal(0, 0.1),
                    "farm_size_ha": rng.uniform(2, 8),
                    "baseline_yield": rng.uniform(1, 3),
                    "soil_quality_index": rng.uniform(0.3, 0.9),
                    "historical_rainfall": rng.uniform(800, 1400),
                }
            )
    return pd.DataFrame(rows)


def _panel_to_wide(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (pre/post yields) and long panel with ``treatment_period``."""
    pre = panel[panel["period"] == 0].copy()
    post = panel[panel["period"] == 1].copy()
    wide = pre.merge(
        post[["farm_id", "yield_tpha"]].rename(columns={"yield_tpha": "yield_post_intervention"}),
        on="farm_id",
    )
    wide = wide.rename(columns={"yield_tpha": "yield_pre_intervention"})
    wide["received_intervention"] = wide["treated"]

    long_panel = panel.copy()
    long_panel["treatment_period"] = np.where(long_panel["treated"] == 1, 1, np.nan)
    long_panel = long_panel.rename(columns={"yield_tpha": "yield"})
    return wide, long_panel


def run_causal_evaluation(panel: pd.DataFrame) -> dict[str, Any]:
    """PSM → balance → DiD → event-study parallel-trends test."""
    wide, long_panel = _panel_to_wide(panel)

    covariates = [c for c in DEFAULT_COVARIATES if c in wide.columns]
    if not covariates:
        covariates = [
            c
            for c in wide.columns
            if c
            not in {
                "farm_id",
                "treated",
                "received_intervention",
                "yield_pre_intervention",
                "yield_post_intervention",
                "role",
            }
        ]

    matched = propensity_score_match(
        wide,
        treatment_col="received_intervention",
        covariate_cols=covariates,
        id_col="farm_id",
    )

    balance = standardized_mean_differences(
        wide,
        matched,
        covariate_cols=covariates,
        treatment_col="received_intervention",
    )

    did = calculate_did_att(matched)

    es = event_study(long_panel)

    pretrend_p = es.pretrend_pvalue
    if pretrend_p is None:
        pretrend_p = 1.0

    did_comparison = compare_did_methods(
        long_panel,
        methods=["twfe", "csdid", "bjs", "synthdid"],
        unit_col="farm_id",
        time_col="period",
        treat_time_col="treatment_period",
        outcome_col="yield",
        covariate_cols=[c for c in covariates if c in long_panel.columns],
        n_boot=199,
        n_placebo=100,
        write_report=True,
    )
    log.info("DiD method comparison:\n%s", did_comparison.to_string(index=False))

    return {
        "max_smd": float(balance.max_smd_matched),
        "max_smd_unmatched": float(balance.max_smd_unmatched),
        "balance_ok": bool(balance.balance_ok),
        "pretrend_pvalue": float(pretrend_p),
        "parallel_trends_ok": bool(es.parallel_trends_ok),
        "did_att": float(did.att),
        "did_se": float(did.se) if did.se is not None else None,
        "n_pairs": int(did.n_pairs),
        "method": did.method,
        "did_method_comparison": did_comparison.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal evaluation → JSON report")
    parser.add_argument("--panel", type=Path, default=Path("data/raw/farm_panel.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/causal_eval.json"))
    parser.add_argument("--synthetic-n", type=int, default=400, help="Panel size if file missing")
    args = parser.parse_args(argv)

    if args.panel.is_file():
        panel = pd.read_parquet(args.panel)
        log.info("Loaded panel from %s (%d rows)", args.panel, len(panel))
    else:
        log.warning("Panel %s not found; using synthetic balanced panel", args.panel)
        panel = _synthetic_panel(args.synthetic_n, seed=42)

    report = run_causal_evaluation(panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log.info("Wrote causal evaluation to %s", args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
