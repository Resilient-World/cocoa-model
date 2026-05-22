"""Temporal cross-validation: forward chaining and season-aware splits."""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np
import pandas as pd

SeasonLabel = str  # main_crop | mid_crop


def assign_season(dates: pd.Series | np.ndarray) -> np.ndarray:
    """
    West Africa cocoa seasons: main crop Oct–Mar, mid crop May–Jul.

    Returns ``main_crop``, ``mid_crop``, or ``off_season``.
    """
    ts = pd.to_datetime(dates)
    months = ts.month.values
    labels = np.full(len(months), "off_season", dtype=object)
    labels[(months >= 10) | (months <= 3)] = "main_crop"
    labels[(months >= 5) & (months <= 7)] = "mid_crop"
    return labels


def iter_forward_folds(
    years: np.ndarray,
    *,
    min_train_years: int = 3,
    step_years: int = 1,
    max_test_years: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_year_indices, test_year_indices)`` into unique sorted years."""
    unique = np.sort(np.unique(years))
    if len(unique) < min_train_years + 1:
        return
    for start in range(min_train_years, len(unique)):
        train_years = unique[:start]
        test_years = unique[start : start + max_test_years]
        if len(test_years) == 0:
            break
        if test_years.min() <= train_years.max():
            continue
        train_idx = np.where(np.isin(years, train_years))[0]
        test_idx = np.where(np.isin(years, test_years))[0]
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


class ForwardChainSplit:
    """Expanding-window time-series CV (no future leakage)."""

    def __init__(
        self,
        *,
        min_train_years: int = 3,
        step_years: int = 1,
        max_test_years: int = 1,
    ) -> None:
        self.min_train_years = min_train_years
        self.step_years = step_years
        self.max_test_years = max_test_years

    def split(
        self,
        years: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        del y
        years = np.asarray(years)
        for train_idx, test_idx in iter_forward_folds(
            years,
            min_train_years=self.min_train_years,
            step_years=self.step_years,
            max_test_years=self.max_test_years,
        ):
            yield train_idx, test_idx

    def get_n_splits(self, years: np.ndarray) -> int:
        return sum(
            1
            for _ in iter_forward_folds(
                np.asarray(years),
                min_train_years=self.min_train_years,
                step_years=self.step_years,
                max_test_years=self.max_test_years,
            )
        )


class SeasonAwareSplit:
    """Hold out one season label per year block for leakage-safe evaluation."""

    def __init__(
        self,
        *,
        holdout_season: SeasonLabel = "main_crop",
        n_folds: int = 2,
    ) -> None:
        self.holdout_season = holdout_season
        self.n_folds = max(2, n_folds)

    def split(
        self,
        dates: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        del y
        seasons = assign_season(dates)
        indices = np.arange(len(dates))
        for fold in range(self.n_folds):
            if fold == 0:
                test_mask = seasons == self.holdout_season
            else:
                test_mask = seasons != self.holdout_season
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            if len(test_idx) and len(train_idx):
                yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_folds
