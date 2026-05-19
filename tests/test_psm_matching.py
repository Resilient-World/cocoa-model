"""Unit tests for propensity score matching."""

import numpy as np
import pandas as pd
import pytest

from analysis.psm_matching import (
    compute_propensity_scores,
    match_nearest_neighbor,
    propensity_score_match,
)


def _synthetic_farms(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    farm_size_ha = rng.uniform(1.0, 10.0, n)
    soil_quality_index = rng.uniform(0.0, 1.0, n)
    historical_rainfall = rng.normal(1200, 200, n)
    baseline_yield = 0.5 * farm_size_ha + 2.0 * soil_quality_index + rng.normal(0, 0.5, n)

    logit = (
        -2.0
        + 0.4 * farm_size_ha
        + 1.5 * soil_quality_index
        + 0.002 * historical_rainfall
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    received_intervention = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "farm_id": [f"farm_{i:03d}" for i in range(n)],
            "received_intervention": received_intervention,
            "farm_size_ha": farm_size_ha,
            "baseline_yield": baseline_yield,
            "soil_quality_index": soil_quality_index,
            "historical_rainfall": historical_rainfall,
        }
    )


def test_propensity_scores_in_unit_interval() -> None:
    df = _synthetic_farms()
    ps = compute_propensity_scores(df)
    assert ps.min() >= 0.0
    assert ps.max() <= 1.0
    assert len(ps) == len(df)


def test_matching_one_to_one_without_replacement() -> None:
    df = _synthetic_farms()
    df = df.copy()
    df["propensity_score"] = compute_propensity_scores(df)
    matched = match_nearest_neighbor(df)

    controls = matched[matched["match_role"] == "control"]
    assert controls["farm_id"].is_unique
    assert matched["match_pair_id"].nunique() == len(controls)


def test_matched_output_has_two_rows_per_pair() -> None:
    df = _synthetic_farms()
    df = df.copy()
    df["propensity_score"] = compute_propensity_scores(df)
    matched = match_nearest_neighbor(df)

    counts = matched.groupby("match_pair_id")["match_role"].agg(list)
    for roles in counts:
        assert sorted(roles) == ["control", "treated"]


def test_end_to_end_propensity_score_match() -> None:
    df = _synthetic_farms()
    matched = propensity_score_match(df)
    assert "propensity_score" in matched.columns
    assert "match_pair_id" in matched.columns
    assert "match_role" in matched.columns
    assert len(matched) == 2 * matched["match_pair_id"].nunique()
    assert len(matched) <= len(df)


def test_caliper_too_tight_raises() -> None:
    df = _synthetic_farms(n=50)
    df = df.copy()
    df["propensity_score"] = compute_propensity_scores(df)
    with pytest.raises(ValueError, match="No matched pairs"):
        match_nearest_neighbor(df, caliper=1e-9)
