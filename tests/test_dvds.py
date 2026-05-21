"""Tests for :mod:`analysis.dvds`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.dvds import MarginalSensitivityModel, dvds_ate, tipping_point
from data.farm_panel import (
    PSM_COVARIATE_COLS,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_synthetic_panel,
)


def test_marginal_sensitivity_model_tau() -> None:
    msm = MarginalSensitivityModel(1.5)
    assert msm.tau == pytest.approx(1.5 / 2.5)
    with pytest.raises(ValueError):
        MarginalSensitivityModel(0.9)


def test_dvds_binary_monotone_width_in_lambda() -> None:
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(size=n)
    z = (rng.random(n) < 0.35).astype(int)
    y = (rng.random(n) < 0.45).astype(float)
    df = pd.DataFrame({"received_intervention": z, "y": y, "x1": x1, "x2": rng.normal(size=n)})
    w1 = dvds_ate(
        df,
        treatment_col="received_intervention",
        outcome_col="y",
        covariate_cols=["x1", "x2"],
        lambda_=1.2,
        n_folds=3,
        random_state=0,
    )
    w2 = dvds_ate(
        df,
        treatment_col="received_intervention",
        outcome_col="y",
        covariate_cols=["x1", "x2"],
        lambda_=2.0,
        n_folds=3,
        random_state=0,
    )
    assert w2.ate_upper - w2.ate_lower >= w1.ate_upper - w1.ate_lower - 1e-6
    assert w1.ate_ci_lower <= w1.ate_lower <= w1.ate_upper <= w1.ate_ci_upper


def test_dvds_farm_panel_contains_synthetic_att() -> None:
    true_att = 0.35
    panel = join_biotic(join_climate(load_synthetic_panel(n_farms=300, true_att=true_att, seed=2)))
    snap = farm_level_snapshot(panel, treatment_year=4)
    snap["yield_delta"] = snap["yield_post_intervention"] - snap["yield_pre_intervention"]
    covs = [c for c in PSM_COVARIATE_COLS if c in snap.columns]
    work = snap.dropna(subset=["yield_delta", *covs])
    res = dvds_ate(
        work,
        treatment_col="received_intervention",
        outcome_col="yield_delta",
        covariate_cols=covs,
        lambda_=2.0,
        n_folds=3,
        random_state=2,
    )
    assert res.ate_lower <= true_att <= res.ate_upper
    assert res.n == len(work)
    assert res.nuisance_diagnostics["outcome_type"] == "continuous"


def test_tipping_point_finite() -> None:
    rng = np.random.default_rng(3)
    n = 350
    x = rng.normal(size=(n, 2))
    z = (rng.random(n) < 1 / (1 + np.exp(-(x[:, 0] + x[:, 1])))).astype(int)
    y = z * 0.4 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"received_intervention": z, "y": y, "a": x[:, 0], "b": x[:, 1]})
    tp = tipping_point(
        df,
        treatment_col="received_intervention",
        outcome_col="y",
        covariate_cols=["a", "b"],
        n_folds=3,
        random_state=3,
    )
    assert 1.0 <= tp <= 10.0
