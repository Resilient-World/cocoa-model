"""Tests for temporal CV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation.temporal_cv import (
    SeasonAwareSplit,
    assign_season,
    iter_forward_folds,
)


def test_forward_chain_no_leakage() -> None:
    years = np.array([2018, 2018, 2019, 2019, 2020, 2021, 2022, 2023])
    for train_idx, test_idx in iter_forward_folds(years, min_train_years=2, max_test_years=1):
        assert years[test_idx].min() > years[train_idx].max()


def test_assign_season_main_crop() -> None:
    dates = pd.to_datetime(["2020-11-01", "2020-06-15", "2020-01-01"])
    seasons = assign_season(dates)
    assert seasons[0] == "main_crop"
    assert seasons[1] == "mid_crop"
    assert seasons[2] == "main_crop"


def test_season_aware_split() -> None:
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    splitter = SeasonAwareSplit()
    folds = list(splitter.split(dates.to_numpy()))
    assert len(folds) >= 1
