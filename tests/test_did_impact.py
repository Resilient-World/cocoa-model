"""Unit tests for DiD ATT and avoided revenue calculations."""

import numpy as np
import pandas as pd
import pytest

from analysis.did_impact import calculate_avoided_revenue_loss, calculate_did_att
from analysis.psm_matching import propensity_score_match


def _matched_panel_with_yields() -> pd.DataFrame:
    n = 120
    rng = np.random.default_rng(0)
    farm_size_ha = rng.uniform(2.0, 8.0, n)
    soil_quality_index = rng.uniform(0.0, 1.0, n)
    historical_rainfall = rng.normal(1200, 150, n)
    baseline_yield = rng.normal(2.0, 0.4, n)
    logit = -1.5 + 0.3 * farm_size_ha + soil_quality_index
    received_intervention = (rng.random(n) < (1 / (1 + np.exp(-logit)))).astype(int)

    df = pd.DataFrame(
        {
            "farm_id": [f"f{i}" for i in range(n)],
            "received_intervention": received_intervention,
            "farm_size_ha": farm_size_ha,
            "baseline_yield": baseline_yield,
            "soil_quality_index": soil_quality_index,
            "historical_rainfall": historical_rainfall,
            "yield_pre_intervention": baseline_yield,
        }
    )
    matched = propensity_score_match(df)
    treated = matched["match_role"] == "treated"
    matched["yield_post_intervention"] = matched["yield_pre_intervention"].copy()
    matched.loc[treated, "yield_post_intervention"] += 0.5
    matched.loc[~treated, "yield_post_intervention"] += 0.1
    return matched


def test_calculate_did_att_known_effect() -> None:
    matched = pd.DataFrame(
        {
            "match_pair_id": [0, 0, 1, 1],
            "match_role": ["treated", "control", "treated", "control"],
            "farm_size_ha": [5.0, 5.0, 3.0, 3.0],
            "yield_pre_intervention": [2.0, 2.0, 1.0, 1.0],
            "yield_post_intervention": [3.0, 2.2, 2.0, 1.1],
        }
    )
    result = calculate_did_att(matched)
    assert result.n_pairs == 2
    assert result.treated_change_mean == 1.0
    assert result.control_change_mean == pytest.approx(0.15, abs=1e-9)
    assert result.att == pytest.approx(0.85, abs=1e-9)


def test_calculate_did_att_on_psm_output() -> None:
    matched = _matched_panel_with_yields()
    result = calculate_did_att(matched)
    assert result.n_pairs > 0
    assert result.att > 0


def test_calculate_avoided_revenue_loss() -> None:
    matched = pd.DataFrame(
        {
            "match_pair_id": [0, 0],
            "match_role": ["treated", "control"],
            "farm_size_ha": [10.0, 10.0],
            "received_intervention": [1, 0],
        }
    )
    att = 0.5
    price = 3000.0
    result = calculate_avoided_revenue_loss(att, matched, cocoa_price_usd=price)
    assert result.n_treated_farms == 1
    assert result.total_avoided_revenue_usd == pytest.approx(0.5 * 10.0 * 3000.0)
    assert result.per_farm_revenue_usd.iloc[0] == pytest.approx(15000.0)


def test_avoided_revenue_sums_treated_cohort() -> None:
    matched = pd.DataFrame(
        {
            "match_pair_id": [0, 0, 1, 1],
            "match_role": ["treated", "control", "treated", "control"],
            "farm_size_ha": [2.0, 2.0, 4.0, 4.0],
            "received_intervention": [1, 0, 1, 0],
        }
    )
    att = 1.0
    price = 100.0
    result = calculate_avoided_revenue_loss(att, matched, cocoa_price_usd=price)
    assert result.n_treated_farms == 2
    assert result.total_avoided_revenue_usd == pytest.approx((2.0 + 4.0) * 100.0)
