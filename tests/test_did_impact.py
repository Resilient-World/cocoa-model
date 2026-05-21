"""Unit tests for DiD ATT and avoided revenue calculations."""

import numpy as np
import pandas as pd
import pytest

from analysis.did_impact import (
    calculate_avoided_revenue_loss,
    calculate_did_att,
    did_estimator,
    event_study,
)
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
    assert result.se is not None
    assert result.ci_low is not None
    assert result.ci_high is not None
    assert result.ci_low <= result.att <= result.ci_high
    assert result.method == "paired_did_bootstrap"


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


def test_avoided_revenue_propagates_att_ci() -> None:
    matched = pd.DataFrame(
        {
            "match_pair_id": [0, 0, 1, 1],
            "match_role": ["treated", "control", "treated", "control"],
            "farm_size_ha": [2.0, 2.0, 4.0, 4.0],
            "received_intervention": [1, 0, 1, 0],
        }
    )
    result = calculate_avoided_revenue_loss(
        1.0,
        matched,
        cocoa_price_usd=100.0,
        att_ci=(0.5, 1.5),
    )
    assert result.total_avoided_revenue_ci_low_usd == pytest.approx((2.0 + 4.0) * 0.5 * 100.0)
    assert result.total_avoided_revenue_ci_high_usd == pytest.approx((2.0 + 4.0) * 1.5 * 100.0)


def test_event_study_runs() -> None:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for farm in ("t1", "t2", "c1", "c2", "c3"):
        treatment_period = 3.0 if farm.startswith("t") else np.nan
        for period in range(8):
            bump = 0.15 * max(0, period - 3) if farm.startswith("t") else 0.0
            rows.append(
                {
                    "farm_id": farm,
                    "period": period,
                    "treatment_period": treatment_period,
                    "yield": 2.0 + bump + rng.normal(0, 0.05),
                }
            )
    panel = pd.DataFrame(rows)
    result = event_study(panel, lead_window=2, lag_window=2)
    assert not result.leads_lags.empty
    assert "period" in result.leads_lags.columns
    assert isinstance(result.parallel_trends_ok, bool)


# ---------------------------------------------------------------------------
# Synthetic-truth tests: estimator must recover known τ within 2 SEs.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402  (already imported above; safe to keep)


def _simulate_matched_panel(
    n_pairs: int = 500,
    true_att: float = 0.4,
    sigma: float = 0.3,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pairs):
        pre_t = rng.normal(2.0, 0.3)
        pre_c = pre_t + rng.normal(0.0, 0.05)  # well-matched on baseline
        common_shock = rng.normal(0.1, sigma)
        post_t = pre_t + common_shock + true_att + rng.normal(0, sigma)
        post_c = pre_c + common_shock + rng.normal(0, sigma)
        rows.append((pid, "treated", 5.0, pre_t, post_t, 1))
        rows.append((pid, "control", 5.0, pre_c, post_c, 0))
    return pd.DataFrame(
        rows,
        columns=[
            "match_pair_id",
            "match_role",
            "farm_size_ha",
            "yield_pre_intervention",
            "yield_post_intervention",
            "received_intervention",
        ],
    )


def test_did_recovers_known_att_within_2_se() -> None:
    panel = _simulate_matched_panel(n_pairs=500, true_att=0.4, seed=11)
    result = calculate_did_att(panel, n_boot=500, random_state=11)
    assert result.se is not None and result.se > 0
    assert abs(result.att - 0.4) < 2 * result.se
    assert result.ci_low < 0.4 < result.ci_high
    assert result.p_value is not None and result.p_value < 0.05


def test_did_null_effect_ci_covers_zero() -> None:
    panel = _simulate_matched_panel(n_pairs=500, true_att=0.0, seed=23)
    result = calculate_did_att(panel, n_boot=500, random_state=23)
    assert result.ci_low < 0.0 < result.ci_high


def test_avoided_revenue_ci_propagation() -> None:
    panel = _simulate_matched_panel(n_pairs=200, true_att=0.5, seed=5)
    did = calculate_did_att(panel, n_boot=300, random_state=5)
    rev = calculate_avoided_revenue_loss(
        did.att,
        panel,
        cocoa_price_usd=3000.0,
        att_ci=(did.ci_low, did.ci_high),
    )
    assert rev.total_avoided_revenue_ci_low_usd is not None
    assert rev.total_avoided_revenue_ci_high_usd is not None
    assert (
        rev.total_avoided_revenue_ci_low_usd
        < rev.total_avoided_revenue_usd
        < rev.total_avoided_revenue_ci_high_usd
    )


def test_did_estimator_csdid_routes() -> None:
    rows = []
    for u in range(30):
        g = float((u % 3) + 1)
        for t in range(4):
            rows.append({
                "farm_id": f"u{u}",
                "period": t,
                "treatment_period": g,
                "yield": 2.0 + 1.5 * (t >= g) + 0.01 * u,
            })
    panel = pd.DataFrame(rows)
    res = did_estimator(panel, method="csdid", n_boot=50, random_state=0)
    assert res.method == "csdid_simple_att"
    assert res.att > 0


def test_did_estimator_synthdid_raises() -> None:
    panel = pd.DataFrame({
        "farm_id": ["a"],
        "period": [0],
        "treatment_period": [np.nan],
        "yield": [1.0],
    })
    with pytest.raises(NotImplementedError):
        did_estimator(panel, method="synthdid")


def test_staggered_deprecation_warning() -> None:
    rows = []
    for u, g in enumerate([1, 2, 3, 4]):
        rows.append({
            "farm_id": f"f{u}",
            "match_pair_id": u,
            "match_role": "treated",
            "yield_pre_intervention": 1.0,
            "yield_post_intervention": 2.0,
            "treatment_period": float(g),
        })
        rows.append({
            "farm_id": f"c{u}",
            "match_pair_id": u,
            "match_role": "control",
            "yield_pre_intervention": 1.0,
            "yield_post_intervention": 1.1,
            "treatment_period": np.nan,
        })
    wide = pd.DataFrame(rows)
    with pytest.warns(DeprecationWarning, match="Staggered"):
        calculate_did_att(wide, unit_col="farm_id", treat_time_col="treatment_period")
