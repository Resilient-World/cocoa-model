"""Tests for farm panel loading and causal ATT recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.psm_matching import aipw_estimator, propensity_score_match, standardized_mean_differences
from analysis.sensitivity import e_value, rosenbaum_bounds
from data.farm_panel import (
    FARM_PANEL_COLUMNS,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_synthetic_panel,
    treatment_year_index,
)


def test_synthetic_panel_schema() -> None:
    panel = load_synthetic_panel(n_farms=50, n_years=6, treatment_year=3, seed=0)
    assert set(FARM_PANEL_COLUMNS).issubset(panel.columns)
    assert len(panel) == 50 * 6


def test_join_climate_and_biotic() -> None:
    panel = load_synthetic_panel(n_farms=30, n_years=5, treatment_year=2, seed=1)
    out = join_biotic(join_climate(panel))
    assert "tmean_annual" in out.columns
    assert "biotic_total_loss_fraction" in out.columns
    assert "cohort_phase" in out.columns


def test_aipw_recovers_synthetic_att_within_two_se() -> None:
    true_att = 0.35
    panel = load_synthetic_panel(
        n_farms=800,
        n_years=8,
        treatment_year=4,
        true_att=true_att,
        seed=7,
    )
    panel = join_biotic(join_climate(panel))
    snapshot = farm_level_snapshot(panel, treatment_year=4)
    from data.farm_panel import PSM_COVARIATE_COLS

    covs = [c for c in PSM_COVARIATE_COLS if c in snapshot.columns]

    matched = propensity_score_match(
        snapshot,
        k=1,
        caliper_scale="logit",
        covariate_cols=covs,
        trim_overlap=True,
        random_state=7,
    )
    balance = standardized_mean_differences(snapshot, matched, covariate_cols=covs)
    assert balance.max_smd_matched <= balance.max_smd_unmatched + 1e-9

    result = aipw_estimator(
        snapshot,
        outcome_col="yield_tonnes_per_ha",
        covariate_cols=covs,
        n_folds=5,
        random_state=7,
    )
    assert result.att_se > 0
    assert abs(result.att - true_att) <= 2.0 * result.att_se


def test_treatment_year_index() -> None:
    panel = load_synthetic_panel(n_farms=20, n_years=6, treatment_year=3, seed=2)
    assert treatment_year_index(panel) == 3


def test_rosenbaum_and_evalue_smoke() -> None:
    from analysis.did_impact import calculate_did_att
    from data.farm_panel import attach_pre_post_to_matched

    panel = load_synthetic_panel(n_farms=400, n_years=6, treatment_year=3, seed=3)
    snapshot = farm_level_snapshot(panel, treatment_year=3)
    from data.farm_panel import PSM_COVARIATE_COLS

    covs = [c for c in PSM_COVARIATE_COLS if c in snapshot.columns][:4]
    matched = propensity_score_match(snapshot, k=1, caliper_scale="logit", covariate_cols=covs)
    matched = attach_pre_post_to_matched(matched, snapshot)
    did = calculate_did_att(matched, n_boot=200, random_state=3)
    bounds = rosenbaum_bounds(matched)
    assert "gamma" in bounds.columns
    ev = e_value(did.att, did.ci_low or did.att - 0.1, outcome_sd=0.5)
    assert ev.point_e_value >= 1.0


def test_load_real_panel_roundtrip(tmp_path: Path) -> None:
    panel = load_synthetic_panel(n_farms=10, n_years=4, treatment_year=2, seed=4)
    path = tmp_path / "panel.parquet"
    panel[list(FARM_PANEL_COLUMNS)].to_parquet(path, index=False)
    from data.farm_panel import load_real_panel

    loaded = load_real_panel(path)
    assert len(loaded) == len(panel)
