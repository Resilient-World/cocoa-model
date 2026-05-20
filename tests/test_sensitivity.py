"""Tests for :mod:`analysis.sensitivity`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.did_impact import calculate_did_att
from analysis.psm_matching import propensity_score_match
from analysis.sensitivity import (
    e_value,
    negative_control_outcome_test,
    rosenbaum_bounds,
    rosenbaum_gamma_at_alpha,
)


def _matched_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 80
    df = pd.DataFrame(
        {
            "farm_id": [f"f{i:03d}" for i in range(n)],
            "received_intervention": (rng.random(n) < 0.4).astype(int),
            "farm_size_ha": rng.uniform(2, 10, n),
            "baseline_yield": rng.uniform(1, 3, n),
            "soil_quality_index": rng.uniform(0.2, 0.9, n),
            "historical_rainfall": rng.normal(1200, 150, n),
            "yield_pre_intervention": rng.uniform(1.0, 2.0, n),
        }
    )
    df["yield_post_intervention"] = (
        df["yield_pre_intervention"]
        + 0.2 * df["received_intervention"]
        + rng.normal(0, 0.1, n)
    )
    return propensity_score_match(df)


def test_rosenbaum_bounds_monotone_in_gamma() -> None:
    matched = _matched_panel()
    bounds = rosenbaum_bounds(matched, gamma_grid=[1.0, 1.25, 1.5, 2.0, 2.5])
    assert list(bounds.columns) == ["gamma", "p_value_upper"]
    assert bounds["p_value_upper"].is_monotonic_increasing


def test_rosenbaum_bounds_outcome_col() -> None:
    matched = _matched_panel()
    matched = matched.copy()
    matched["pair_yield_change"] = matched["yield_post_intervention"] - matched["yield_pre_intervention"]
    bounds = rosenbaum_bounds(matched, "pair_yield_change", gamma_grid=[1.0, 2.0])
    assert len(bounds) == 2
    assert (bounds["p_value_upper"] >= 0).all()


def test_rosenbaum_gamma_at_alpha() -> None:
    bounds = pd.DataFrame({"gamma": [1.0, 1.5, 2.0], "p_value_upper": [0.01, 0.04, 0.12]})
    assert rosenbaum_gamma_at_alpha(bounds, alpha=0.05) == 2.0
    assert rosenbaum_gamma_at_alpha(bounds, alpha=0.03) == 1.5


def test_e_value_from_se() -> None:
    ev = e_value(0.35, 0.08, outcome_sd=0.5)
    assert ev.point_e_value >= 1.0
    assert ev.ci_e_value >= 1.0
    assert ev.ci_low == pytest.approx(0.35 - 1.96 * 0.08)


def test_e_value_null_effect() -> None:
    ev = e_value(0.0, 0.1)
    assert ev.point_e_value == 1.0
    assert ev.ci_e_value == 1.0


def test_negative_control_outcome_test() -> None:
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "received_intervention": (rng.random(n) < 0.5).astype(int),
            "soil_quality_index": rng.uniform(0.2, 0.9, n),
        }
    )
    result = negative_control_outcome_test(df, "soil_quality_index")
    assert result.falsification_pass is True
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_treated + result.n_control == n


def test_negative_control_detects_spurious_association() -> None:
    rng = np.random.default_rng(2)
    n = 120
    treated = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame(
        {
            "received_intervention": treated,
            "nco_bad": treated * 2.0 + rng.normal(0, 0.05, n),
        }
    )
    result = negative_control_outcome_test(df, "nco_bad", alpha=0.05)
    assert result.falsification_pass is False
    assert result.p_value < 0.05


def test_rosenbaum_consistent_with_did_direction() -> None:
    matched = _matched_panel(seed=42)
    did = calculate_did_att(matched)
    bounds = rosenbaum_bounds(matched, gamma_grid=np.linspace(1.0, 2.0, 5))
    if did.att > 0:
        assert bounds.loc[0, "p_value_upper"] < 0.5
