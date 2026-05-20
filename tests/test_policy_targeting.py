"""Tests for :mod:`analysis.policy_targeting`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.heterogeneity import CATEResult
from analysis.policy_targeting import policy_value_curve, rank_farms_by_uplift


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

