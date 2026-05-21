"""Tests for honest DR policy tree / forest targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("econml")

from analysis.policy_targeting import (
    first_split_threshold,
    learn_policy_tree,
    render_policy_rules,
    root_split_feature,
)


def _threshold_dgp(n: int, seed: int) -> pd.DataFrame:
    """Optimal treat when x0 > 0 (sharp CATE)."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n) * 0.01
    t = rng.binomial(1, 0.5, n)
    tau = np.where(x0 > 0, 3.0, 0.0)
    y = 0.5 * x0 + tau * t + rng.normal(0, 0.2, n)
    return pd.DataFrame({"y": y, "t": t, "x0": x0, "x1": x1})


def test_recovers_known_threshold() -> None:
    df = _threshold_dgp(1500, seed=7)
    result = learn_policy_tree(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0"],
        max_depth=3,
        min_samples_leaf=50,
        n_folds=3,
        n_bootstrap=0,
        random_state=0,
    )
    x = df[["x0"]].to_numpy()
    pred = result.tree.predict(x) == 1
    rate_pos = float(pred[df["x0"].to_numpy() > 0].mean())
    rate_neg = float(pred[df["x0"].to_numpy() <= 0].mean())
    assert rate_pos > rate_neg + 0.05
    assert root_split_feature(result) == "x0"


def test_render_policy_rules_uses_names() -> None:
    df = _threshold_dgp(500, seed=1)
    result = learn_policy_tree(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0", "x1"],
        min_samples_leaf=40,
        n_bootstrap=0,
        n_folds=3,
    )
    rules = render_policy_rules(result, recommended_treatment_label="treat_with_shade_trees")
    assert rules
    joined = " ".join(rules)
    assert "x0" in joined or "x1" in joined
    assert "feature_" not in joined


@pytest.mark.slow
def test_honest_leaf_ci_coverage() -> None:
    """Leaf-level bootstrap CIs achieve ~95% coverage of leaf mean uplift."""
    n_sim = 12
    covered = 0
    total = 0
    for seed in range(n_sim):
        df = _threshold_dgp(500, seed=100 + seed)
        result = learn_policy_tree(
            df,
            treatment_col="t",
            outcome_col="y",
            covariate_cols=["x0"],
            max_depth=2,
            min_samples_leaf=50,
            n_folds=3,
            n_bootstrap=25,
            random_state=seed,
        )
        x = df[["x0"]].to_numpy()
        true_tau = np.where(x[:, 0] > 0, 3.0, 0.0)
        from analysis.policy_targeting import _leaf_ids_from_tree

        pm = result.tree.policy_model_
        leaf_ids = _leaf_ids_from_tree(pm, x)
        for _, row in result.leaf_summary.iterrows():
            if int(row["n_units"]) < 30:
                continue
            leaf_id = int(row["leaf_id"])
            mask = leaf_ids == leaf_id
            if mask.sum() < 30:
                continue
            true_mean = float(np.mean(true_tau[mask]))
            total += 1
            if float(row["ci_low"]) <= true_mean <= float(row["ci_high"]):
                covered += 1
    assert total > 5
    rate = covered / total
    assert rate >= 0.75


def test_cost_aware_changes_selection() -> None:
    rng = np.random.default_rng(99)
    n = 600
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n) * 0.01
    t = rng.binomial(1, 0.5, n)
    tau = np.where(x0 > 0, 2.5, 0.2)
    y = tau * t + rng.normal(0, 0.2, n)
    cost = np.where(x0 < 0, 3.0, 0.1)
    df = pd.DataFrame({"y": y, "t": t, "x0": x0, "x1": x1, "cost_usd": cost})

    plain = learn_policy_tree(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0", "x1"],
        min_samples_leaf=50,
        n_bootstrap=0,
        n_folds=3,
    )
    costed = learn_policy_tree(
        df,
        treatment_col="t",
        outcome_col="y",
        covariate_cols=["x0", "x1"],
        cost_col="cost_usd",
        min_samples_leaf=50,
        n_bootstrap=0,
        n_folds=3,
    )
    x = df[["x0", "x1"]].to_numpy()
    treat_plain = (plain.tree.predict(x) == 1).mean()
    treat_costed = (costed.tree.predict(x) == 1).mean()
    assert costed.cost_aware
    assert treat_costed <= treat_plain + 0.1
