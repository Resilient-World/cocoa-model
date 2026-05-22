"""NGBoost vs HGB nuisance recovery under heteroscedastic noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ngboost")

from analysis.psm_matching import aipw_estimator


def _heteroscedastic_panel(n: int = 800, *, seed: int = 0) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = -0.5 + 0.8 * x1 + 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-logit))
    treat = (rng.uniform(size=n) < p).astype(int)
    true_att = 0.4
    noise_scale = np.exp(0.5 * x1)
    y = 2.0 + true_att * treat + rng.normal(scale=noise_scale)
    df = pd.DataFrame(
        {
            "farm_id": np.arange(n),
            "received_intervention": treat,
            "yield_delta": y,
            "x1": x1,
            "x2": x2,
        }
    )
    return df, true_att


def test_ngboost_aipw_ci_contains_truth() -> None:
    df, true_att = _heteroscedastic_panel()
    res = aipw_estimator(
        df,
        outcome_col="yield_delta",
        covariate_cols=["x1", "x2"],
        nuisance_estimator="ngboost",
        n_folds=3,
        random_state=1,
    )
    assert res.ate_ci_low <= true_att <= res.ate_ci_high


def test_ngboost_se_not_worse_than_hgb() -> None:
    df, _ = _heteroscedastic_panel(seed=1)
    hgb = aipw_estimator(
        df,
        outcome_col="yield_delta",
        covariate_cols=["x1", "x2"],
        nuisance_estimator="hgb",
        n_folds=3,
        random_state=2,
    )
    ngb = aipw_estimator(
        df,
        outcome_col="yield_delta",
        covariate_cols=["x1", "x2"],
        nuisance_estimator="ngboost",
        n_folds=3,
        random_state=2,
    )
    assert ngb.ate_se <= hgb.ate_se * 1.25
