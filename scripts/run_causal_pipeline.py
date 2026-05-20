#!/usr/bin/env python3
"""
End-to-end farm panel causal pipeline: PSM → balance → AIPW → DiD → sensitivity report.

Example::

    python scripts/run_causal_pipeline.py --synthetic
    python scripts/run_causal_pipeline.py --panel data/raw/farm_panel.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from analysis._report import write_att_report
from analysis.did_impact import calculate_did_att
from analysis.psm_matching import (
    aipw_estimator,
    compute_propensity_scores,
    default_logit_caliper,
    match_nearest_neighbor,
    standardized_mean_differences,
    trim_common_support,
)
from analysis.sensitivity import e_value, rosenbaum_bounds
from data.farm_panel import (
    PSM_COVARIATE_COLS,
    attach_pre_post_to_matched,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_real_panel,
    load_synthetic_panel,
)

logger = logging.getLogger(__name__)
DEFAULT_REPORT_DIR = _REPO_ROOT / "reports" / "causal"
DEFAULT_ERA5 = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"


def run_pipeline(
    panel: pd.DataFrame,
    *,
    treatment_year: int = 4,
    era5_zarr_path: Path | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    true_att: float | None = None,
    random_state: int = 42,
) -> int:
    """Execute PSM, balance gate, AIPW, DiD, cross-check, and HTML report."""
    panel = join_climate(panel, era5_zarr_path)
    panel = join_biotic(panel)

    snapshot = farm_level_snapshot(panel, treatment_year=treatment_year)
    covariates = [c for c in PSM_COVARIATE_COLS if c in snapshot.columns]

    work = snapshot.copy()
    work["propensity_score"] = compute_propensity_scores(
        work,
        covariate_cols=covariates,
        random_state=random_state,
    )
    work = trim_common_support(work)
    base_caliper = default_logit_caliper(work["propensity_score"].to_numpy())
    balance_covs = [c for c in covariates if c != "farm_size_ha"] or list(covariates)

    matched = None
    balance = None
    for mult in (1.0, 0.85, 0.7, 0.55, 0.45):
        try:
            candidate = match_nearest_neighbor(
                work,
                k=1,
                caliper=base_caliper * mult,
                caliper_scale="logit",
            )
        except ValueError:
            continue
        rep = standardized_mean_differences(snapshot, candidate, covariate_cols=balance_covs)
        matched = candidate
        balance = rep
        if rep.balance_ok:
            logger.info("PSM balance OK (caliper mult=%.2f)", mult)
            break

    if matched is None or balance is None:
        logger.error("PSM produced no matched pairs")
        return 1
    if not balance.balance_ok:
        logger.error(
            "Covariate balance failed: max |SMD|=%.3f (threshold 0.10)",
            balance.max_smd_matched,
        )
        write_att_report(
            report_dir,
            aipw=_empty_aipw_placeholder(),
            did=_empty_did_placeholder(),
            balance=balance,
            att_agreement_ok=False,
            att_agreement_delta=float("nan"),
            rosenbaum=pd.DataFrame(),
            evalue=e_value(0.0, 0.0),
            panel_summary=_panel_summary(panel, snapshot),
            true_att=true_att,
        )
        return 1

    aipw = aipw_estimator(
        snapshot,
        outcome_col="yield_tonnes_per_ha",
        covariate_cols=covariates,
        n_folds=5,
        random_state=random_state,
    )

    matched_did = attach_pre_post_to_matched(matched, snapshot)
    did = calculate_did_att(matched_did, random_state=random_state)

    did_se = float(did.se) if did.se is not None else aipw.att_se
    se_pool = float(np.sqrt(aipw.att_se**2 + did_se**2))
    att_delta = abs(aipw.att - did.att)
    att_agreement_ok = att_delta <= se_pool

    outcome_sd = float(snapshot["yield_tonnes_per_ha"].std(ddof=1))
    evalue = e_value(aipw.att, aipw.att_se, outcome_sd=outcome_sd)
    rosenbaum = rosenbaum_bounds(matched_did)

    report_path = write_att_report(
        report_dir,
        aipw=aipw,
        did=did,
        balance=balance,
        att_agreement_ok=att_agreement_ok,
        att_agreement_delta=att_delta,
        rosenbaum=rosenbaum,
        evalue=evalue,
        panel_summary=_panel_summary(panel, snapshot),
        true_att=true_att,
    )

    logger.info("AIPW ATT=%.4f (SE=%.4f)", aipw.att, aipw.att_se)
    logger.info("DiD ATT=%.4f (SE=%s)", did.att, did.se)
    logger.info("ATT agreement within 1 SE: %s", att_agreement_ok)
    logger.info("Report: %s", report_path)

    if not att_agreement_ok:
        logger.error("AIPW vs DiD ATT differ by %.4f (> 1 SE=%.4f)", att_delta, se_pool)
        return 1
    return 0


def _panel_summary(panel: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    return {
        "n_rows": len(panel),
        "n_farms": panel["farm_id"].nunique(),
        "n_treated_farms": int((snapshot["received_intervention"] == 1).sum()),
    }


def _empty_aipw_placeholder():
    from analysis.psm_matching import AIPWResult

    return AIPWResult(
        att=0.0,
        att_se=0.0,
        att_ci_low=0.0,
        att_ci_high=0.0,
        ate=0.0,
        ate_se=0.0,
        ate_ci_low=0.0,
        ate_ci_high=0.0,
        n=0,
        n_treated=0,
        n_folds=5,
    )


def _empty_did_placeholder():
    from analysis.did_impact import DiDResult

    return DiDResult(att=0.0, treated_change_mean=0.0, control_change_mean=0.0, n_pairs=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Farm panel causal ATT pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic panel (ATT=0.35)")
    parser.add_argument("--panel", type=Path, default=None, help="Real panel parquet path")
    parser.add_argument("--n-farms", type=int, default=5000)
    parser.add_argument("--n-years", type=int, default=8)
    parser.add_argument("--treatment-year", type=int, default=4)
    parser.add_argument("--true-att", type=float, default=0.35)
    parser.add_argument("--era5-zarr", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.synthetic:
        panel = load_synthetic_panel(
            n_farms=args.n_farms,
            n_years=args.n_years,
            treatment_year=args.treatment_year,
            true_att=args.true_att,
            seed=args.seed,
        )
        true_att = args.true_att
    elif args.panel is not None:
        panel = load_real_panel(args.panel)
        true_att = None
    else:
        parser.error("Specify --synthetic or --panel PATH")

    era5 = args.era5_zarr if args.era5_zarr.is_dir() else None
    return run_pipeline(
        panel,
        treatment_year=args.treatment_year,
        era5_zarr_path=era5,
        report_dir=args.report_dir,
        true_att=true_att,
        random_state=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
