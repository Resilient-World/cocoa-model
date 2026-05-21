"""R-Learner CATE (econml ForestDRLearner) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.heterogeneity import RLearnerCATE, estimate_cate

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _simulate_cate_dgp(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    logits = 0.4 * x0 - 0.3 * x1
    p = 1.0 / (1.0 + np.exp(-logits))
    t = rng.binomial(1, p, n).astype(int)
    tau = 0.5 + 1.0 * x0
    mu = 1.0 + 0.8 * x0 - 0.2 * x1
    y = mu + tau * t + rng.normal(0, 0.65, n)
    df = pd.DataFrame({"y": y, "t": t, "x0": x0, "x1": x1})
    return df, tau


def test_rlearner_rmse_recovery() -> None:
    df, tau_true = _simulate_cate_dgp(n=4000, seed=1)
    est = RLearnerCATE(n_estimators=200, n_folds=5, random_state=1, min_samples_leaf=5)
    est.fit(df, "t", "y", ["x0", "x1"])
    x = df[["x0", "x1"]].to_numpy()
    eff = est.effect(x)
    rmse = float(np.sqrt(np.mean((eff.point - tau_true) ** 2)))
    assert rmse < 0.16, f"RMSE={rmse}"


def test_rlearner_ci_coverage() -> None:
    df, tau_true = _simulate_cate_dgp(n=4000, seed=10)
    est = RLearnerCATE(n_estimators=200, n_folds=5, random_state=10, min_samples_leaf=5)
    est.fit(df, "t", "y", ["x0", "x1"])
    x = df[["x0", "x1"]].to_numpy()
    eff = est.effect(x, alpha=0.05)
    covered = (eff.ci_low <= tau_true) & (tau_true <= eff.ci_high)
    rate = float(covered.mean())
    assert 0.92 <= rate <= 0.98, f"coverage={rate}"


def test_rlearner_correlation_with_truth() -> None:
    df, tau_true = _simulate_cate_dgp(n=2000, seed=12)
    res = estimate_cate(
        df,
        outcome="y",
        treatment="t",
        covariates=["x0", "x1"],
        method="r_learner",
        n_folds=3,
        n_estimators=100,
        random_state=12,
    )
    corr = float(np.corrcoef(res.tau_hat.to_numpy(), tau_true)[0, 1])
    assert corr > 0.7
