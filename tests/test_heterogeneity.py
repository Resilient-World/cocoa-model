"""Tests for CATE estimators in :mod:`analysis.heterogeneity`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.heterogeneity import estimate_cate


def _simulate_heterogeneous_dgp(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Propensity depends on x1/x2 (overlap maintained)
    logits = 0.4 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = rng.binomial(1, p, n).astype(int)

    tau = 0.5 + x1  # true heterogeneous effect
    mu = 1.0 + 0.8 * x1 - 0.2 * x2
    y = mu + tau * t + rng.normal(0, 1.0, n)

    df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
    return df, tau


def test_estimate_cate_rlearner_smoke() -> None:
    df, tau = _simulate_heterogeneous_dgp(n=1200, seed=1)
    res = estimate_cate(
        df,
        outcome="y",
        treatment="t",
        covariates=["x1", "x2"],
        method="r_learner",
        n_folds=3,
        n_estimators=100,
        random_state=1,
    )
    assert len(res.tau_hat) == len(df)
    assert res.se.notna().any()
    corr = float(np.corrcoef(res.tau_hat.to_numpy(), tau)[0, 1])
    assert corr > 0.5


def test_estimate_cate_causal_forest_smoke() -> None:
    df, tau = _simulate_heterogeneous_dgp(n=1200, seed=2)
    res = estimate_cate(
        df,
        outcome="y",
        treatment="t",
        covariates=["x1", "x2"],
        method="causal_forest",
        n_folds=3,
        n_estimators=100,
        random_state=2,
    )
    tau_hat = res.tau_hat.to_numpy()
    mse_model = float(np.mean((tau_hat - tau) ** 2))
    mse_const = float(np.mean((np.full_like(tau, tau.mean()) - tau) ** 2))
    assert mse_model < mse_const

