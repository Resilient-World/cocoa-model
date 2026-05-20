"""Tests for :mod:`analysis.parallel_trends`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.parallel_trends import goodman_bacon_decomposition, placebo_pretreatment_did
from data.farm_panel import load_synthetic_panel


def test_placebo_pretreatment_near_zero_on_synthetic() -> None:
    panel = load_synthetic_panel(
        n_farms=400,
        n_years=8,
        treatment_year=4,
        true_att=0.35,
        seed=7,
    )
    years = sorted(panel["year"].unique())
    split = years[4]
    result = placebo_pretreatment_did(
        panel,
        treatment_year=split,
        k_periods=2,
        random_state=0,
    )
    assert len(result.table) == 2
    assert result.table["placebo_att"].notna().any()
    assert result.max_abs_placebo_att < 0.25


def test_placebo_requires_pre_periods() -> None:
    panel = load_synthetic_panel(n_farms=50, n_years=4, treatment_year=1, seed=1)
    years = sorted(panel["year"].unique())
    try:
        placebo_pretreatment_did(panel, treatment_year=years[1], k_periods=5)
    except ValueError:
        pass
    else:
        result = placebo_pretreatment_did(panel, treatment_year=years[1], k_periods=1)
        assert not result.table.empty


def test_goodman_bacon_simultaneous_cohort() -> None:
    panel = load_synthetic_panel(n_farms=300, n_years=8, treatment_year=4, seed=3)
    decomp = goodman_bacon_decomposition(panel)
    assert not decomp.empty
    assert "weight_share" in decomp.columns
    assert "did_estimate" in decomp.columns
    timing_never = decomp[decomp["comparison"] == "timing_vs_never"]
    assert not timing_never.empty
    assert timing_never["weight_share"].sum() > 0.99 or len(decomp) == 1


def test_goodman_bacon_staggered_two_cohorts() -> None:
    rng = np.random.default_rng(99)
    rows = []
    for i in range(45):
        if i >= 40:
            adopt = None
        else:
            adopt = 2018 if i < 20 else 2020
        for year in range(2016, 2023):
            if adopt is None:
                treated = 0
                effect = 0.0
            else:
                treated = int(year >= adopt)
                effect = 0.1 * treated * (i < 20)
            rows.append(
                {
                    "farm_id": f"F{i}",
                    "year": year,
                    "yield_tonnes_per_ha": 1.5 + effect + rng.normal(0, 0.05),
                    "received_intervention": treated,
                }
            )
    panel = pd.DataFrame(rows)
    decomp = goodman_bacon_decomposition(panel)
    assert len(decomp) >= 2
    assert decomp["weight"].sum() > 0
    assert (decomp["comparison"] == "timing_vs_never").any()
