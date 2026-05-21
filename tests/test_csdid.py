"""Tests for Callaway-Sant'Anna staggered DiD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.csdid import ATTGTResult, CallawaySantAnna

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "csdid"
TOL = 0.01


def test_att_gt_skips_when_t_lt_g() -> None:
    df = pd.DataFrame(
        {
            "farm_id": ["a", "a", "b", "b"],
            "period": [0, 1, 0, 1],
            "treatment_period": [1.0, 1.0, np.nan, np.nan],
            "yield": [1.0, 2.0, 1.5, 1.6],
        }
    )
    est = CallawaySantAnna(df, n_boot=50, random_state=0)
    r = est.att_gt(1, 0)
    assert np.isnan(r.att)
    assert r.n_treated == 0


@pytest.mark.slow
def test_mpdta_replication_within_benchmarks() -> None:
    df = pd.read_csv(FIXTURES / "mpdta.csv")
    with (FIXTURES / "mpdta_benchmarks.json").open(encoding="utf-8") as f:
        bench = json.load(f)
    est = CallawaySantAnna(
        df,
        unit_col="countyreal",
        time_col="year",
        treat_time_col="first.treat",
        outcome_col="lemp",
        covariate_cols=["lpop"],
        n_boot=199,
        random_state=42,
    )
    assert abs(est.simple_att().att - bench["simple_att"]) < TOL
    for key, target in bench.items():
        if not key.startswith("att_gt_"):
            continue
        _, g, t = key.split("_")
        got = est.att_gt(int(g), int(t)).att
        assert abs(got - target) < TOL, f"{key}: {got} vs {target}"


def test_event_study_has_bands() -> None:
    df = pd.read_csv(FIXTURES / "mpdta.csv")
    est = CallawaySantAnna(
        df,
        unit_col="countyreal",
        time_col="year",
        treat_time_col="first.treat",
        outcome_col="lemp",
        covariate_cols=["lpop"],
        n_boot=99,
        random_state=1,
    )
    es = est.event_study_aggregation(min_e=0, max_e=3)
    assert not es.leads_lags.empty
    assert {"event_time", "att", "ci_low", "ci_high"}.issubset(es.leads_lags.columns)


def test_group_and_calendar_att() -> None:
    df = pd.read_csv(FIXTURES / "mpdta.csv")
    est = CallawaySantAnna(
        df,
        unit_col="countyreal",
        time_col="year",
        treat_time_col="first.treat",
        outcome_col="lemp",
        n_boot=50,
    )
    gpath = est.group_att(2004)
    assert 2004 in gpath
    cpath = est.calendar_att(2007)
    assert 2007 in cpath


def test_att_gt_result_type() -> None:
    r = ATTGTResult(2004, 2007, -0.1, 0.02, -0.14, -0.06, 10, 100)
    assert r.g == 2004
