"""Tests for :mod:`analysis.policy_targeting`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.heterogeneity import CATEResult
from analysis.policy_targeting import (
    doubly_robust_policy_value,
    optimal_targeting_policy,
    policy_value_curve,
    rank_farms_by_uplift,
    targeting_from_cate,
)


def test_policy_targeting_orders_high_uplift_first() -> None:
    idx = pd.Index(["a", "b", "c", "d"])
    cate = CATEResult(
        tau_hat=pd.Series([0.6, 0.1, 0.4, 0.2], index=idx),
        se=pd.Series([0.3, 0.01, 0.2, 0.05], index=idx),
        ci_low=pd.Series([0.0, 0.0, 0.0, 0.0], index=idx),
        ci_high=pd.Series([1.0, 1.0, 1.0, 1.0], index=idx),
        feature_importances=None,
        method="r_learner",
        n_folds=5,
    )
    areas = pd.Series([1.0, 5.0, 2.0, 1.0], index=idx)
    ranked = rank_farms_by_uplift(
        cate,
        intervention_cost_usd_per_farm=100.0,
        cocoa_price_usd=1000.0,
        farm_areas_ha=areas,
    )
    # Farm b has large area but tiny tau; farm a and c should rank high.
    assert ranked.index[0] in ("a", "c")

    curve = policy_value_curve(ranked, uplift_col="net_uplift_usd")
    assert curve["k"].iloc[0] == 1
    assert np.all(np.diff(curve["cumulative_value"]) >= -1e-9)


def test_optimal_targeting_policy_greedy_budget() -> None:
    tau = pd.Series([2.0, 1.0, 0.5, 3.0], index=["a", "b", "c", "d"])
    costs = pd.Series([100.0, 50.0, 50.0, 200.0], index=tau.index)
    mask = optimal_targeting_policy(tau, costs, budget=250.0)
    assert mask.sum() >= 1
    assert mask.sum() <= len(tau)
    # Highest cate/cost (d: 3/200=0.015, a: 2/100=0.02) should be prioritized
    assert mask[tau.index.get_loc("a")] or mask[tau.index.get_loc("d")]


def test_doubly_robust_policy_value_orders_better_policy() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    tau_true = 0.5 + x0
    y = x0 + tau_true * t + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "t": t, "x0": x0, "x1": x1})
    tau_hat = tau_true + rng.normal(0, 0.1, n)
    good = (tau_hat > np.median(tau_hat)).astype(bool)
    bad = ~good
    v_good = doubly_robust_policy_value(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0", "x1"],
        policy_mask=good,
        tau_hat=tau_hat,
        n_folds=3,
    )
    v_bad = doubly_robust_policy_value(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0", "x1"],
        policy_mask=bad,
        tau_hat=tau_hat,
        n_folds=3,
    )
    assert v_good > v_bad


def test_targeting_from_cate() -> None:
    idx = pd.Index(["a", "b", "c"])
    cate = CATEResult(
        tau_hat=pd.Series([2.0, 0.5, 1.0], index=idx),
        se=pd.Series([0.1, 0.1, 0.1], index=idx),
        ci_low=pd.Series([0.0, 0.0, 0.0], index=idx),
        ci_high=pd.Series([3.0, 3.0, 3.0], index=idx),
    )
    costs = pd.Series([100.0, 50.0, 80.0], index=idx)
    mask, ranked = targeting_from_cate(cate, costs, budget=150.0)
    assert mask.sum() >= 1
    assert "selected" in ranked.columns
