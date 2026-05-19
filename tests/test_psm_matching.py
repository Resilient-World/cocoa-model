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


# ---------------------------------------------------------------------------
# DML / balance / logit-caliper extensions
# ---------------------------------------------------------------------------


def _synthetic_outcome_panel(n: int = 400, true_att: float = 0.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    farm_size_ha = rng.uniform(1.0, 10.0, n)
    soil_quality_index = rng.uniform(0.0, 1.0, n)
    historical_rainfall = rng.normal(1200, 200, n)
    baseline_yield = 0.5 * farm_size_ha + 2.0 * soil_quality_index + rng.normal(0, 0.5, n)
    logit = -2.0 + 0.4 * farm_size_ha + 1.5 * soil_quality_index
    prob = 1.0 / (1.0 + np.exp(-logit))
    received_intervention = (rng.random(n) < prob).astype(int)
    outcome = baseline_yield + true_att * received_intervention + rng.normal(0, 0.15, n)
    return pd.DataFrame(
        {
            "farm_id": [f"farm_{i:03d}" for i in range(n)],
            "received_intervention": received_intervention,
            "farm_size_ha": farm_size_ha,
            "baseline_yield": baseline_yield,
            "soil_quality_index": soil_quality_index,
            "historical_rainfall": historical_rainfall,
            "outcome_yield": outcome,
        }
    )


def test_default_logit_caliper_positive() -> None:
    from analysis.psm_matching import default_logit_caliper

    ps = np.linspace(0.15, 0.85, 50)
    assert default_logit_caliper(ps) > 0


def test_k_nearest_matching_returns_k_controls_per_treated() -> None:
    df = _synthetic_farms(n=80)
    df = df.copy()
    df["propensity_score"] = compute_propensity_scores(df)
    matched = match_nearest_neighbor(df, k=2, with_replacement=True)
    counts = matched.groupby("match_pair_id")["match_role"].value_counts().unstack(fill_value=0)
    assert (counts["treated"] == 1).all()
    assert (counts["control"] == 2).all()


def test_balance_report_structure_and_love_plot() -> None:
    from analysis.psm_matching import (
        default_logit_caliper,
        love_plot_data,
        standardized_mean_differences,
    )

    df = _synthetic_farms(n=200)
    covs = ["farm_size_ha", "baseline_yield", "soil_quality_index", "historical_rainfall"]
    ps = compute_propensity_scores(df)
    caliper = default_logit_caliper(ps.to_numpy())
    matched = propensity_score_match(df, caliper=caliper, caliper_scale="logit")
    report = standardized_mean_differences(df, matched, covariate_cols=covs)
    assert len(report.smd) == len(covs)
    assert report.max_smd_matched >= 0.0
    assert isinstance(report.balance_ok, bool)
    love = love_plot_data(report)
    assert set(love["stage"].unique()) <= {"before", "after"}


def test_aipw_recovers_positive_att() -> None:
    from analysis.psm_matching import aipw_estimator

    rng = np.random.default_rng(1)
    n = 800
    farm_size_ha = rng.uniform(1.0, 10.0, n)
    soil_quality_index = rng.uniform(0.0, 1.0, n)
    historical_rainfall = rng.normal(1200, 200, n)
    baseline_yield = rng.normal(2.0, 0.3, n)
    received_intervention = rng.binomial(1, 0.5, n).astype(int)
    true_att = 1.0
    outcome_yield = baseline_yield + true_att * received_intervention + rng.normal(0, 0.2, n)
    df = pd.DataFrame(
        {
            "farm_id": [f"farm_{i:03d}" for i in range(n)],
            "received_intervention": received_intervention,
            "farm_size_ha": farm_size_ha,
            "baseline_yield": baseline_yield,
            "soil_quality_index": soil_quality_index,
            "historical_rainfall": historical_rainfall,
            "outcome_yield": outcome_yield,
        }
    )
    result = aipw_estimator(df, outcome_col="outcome_yield", n_folds=5, random_state=1)
    assert result.method == "dml_aipw_crossfit"
    assert result.ate > 0.2
    assert result.att > 0.2
    assert result.ate_ci_low > 0
    assert result.att_ci_low > 0
    assert result.n_treated > 0


def test_trim_overlap_reduces_or_preserves_sample() -> None:
    from analysis.psm_matching import trim_common_support

    df = _synthetic_farms(n=100)
    df = df.copy()
    df["propensity_score"] = compute_propensity_scores(df)
    trimmed = trim_common_support(df)
    assert len(trimmed) <= len(df)
    assert len(trimmed) > 0


from analysis.psm_matching import (
    DEFAULT_COVARIATES,
    aipw_estimator,
    default_logit_caliper,
    love_plot_data,
    standardized_mean_differences,
    trim_common_support,
)


def test_smd_improves_after_matching() -> None:
    df = _synthetic_farms(n=400)
    matched = propensity_score_match(df)
    rep = standardized_mean_differences(df, matched, covariate_cols=list(DEFAULT_COVARIATES))
    assert rep.max_smd_matched <= rep.max_smd_unmatched + 1e-9
    assert rep.smd.shape == (4, 3)


def test_logit_caliper_default_is_positive() -> None:
    df = _synthetic_farms(n=300)
    df["propensity_score"] = compute_propensity_scores(df)
    c = default_logit_caliper(df["propensity_score"].to_numpy())
    assert c > 0
    matched = match_nearest_neighbor(df, caliper=c, caliper_scale="logit")
    assert "match_pair_id" in matched.columns


def test_k_to_one_matching_structure() -> None:
    df = _synthetic_farms(n=300)
    df["propensity_score"] = compute_propensity_scores(df)
    matched = match_nearest_neighbor(df, k=2, with_replacement=True)
    counts = matched.groupby("match_pair_id")["match_role"].value_counts().unstack()
    assert (counts["treated"] == 1).all()
    assert (counts["control"] == 2).all()


def test_trim_common_support_keeps_both_groups() -> None:
    df = _synthetic_farms(n=300).copy()
    df["propensity_score"] = compute_propensity_scores(df)
    out = trim_common_support(df)
    assert (out["received_intervention"] == 1).any()
    assert (out["received_intervention"] == 0).any()


def test_dml_aipw_recovers_known_att() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.4 * x1 - 0.3 * x2)))
    a = (rng.random(n) < p).astype(int)
    true_att = 0.8
    y = 1.0 + 0.5 * x1 + 0.7 * x2 + true_att * a + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"farm_id": range(n), "received_intervention": a, "x1": x1, "x2": x2, "y": y})
    r = aipw_estimator(df, outcome_col="y", covariate_cols=["x1", "x2"], n_folds=5)
    assert r.att_se > 0
    assert abs(r.att - true_att) < 3 * r.att_se
    assert r.att_ci_low < true_att < r.att_ci_high
