"""Tests for spatial block CV."""

from __future__ import annotations

import numpy as np
import pytest

from validation.spatial_cv import (
    BufferedLOO,
    SpatialBlockSplit,
    compute_residual_variogram,
    recommend_block_size_km,
    spatial_holdout_mask,
)


def test_checkerboard_folds_disjoint() -> None:
    lats = np.linspace(5, 7, 200)
    lons = np.linspace(-6, -3, 200)
    splitter = SpatialBlockSplit(block_size_km=50.0, n_folds=5, strategy="checkerboard", seed=0)
    train_idx, test_idx = next(splitter.split(lats, lons))
    assert not set(train_idx) & set(test_idx)
    assert len(test_idx) > 0


def test_buffered_loo_excludes_neighbors() -> None:
    lats = np.array([6.0, 6.0, 8.0])
    lons = np.array([-2.0, -2.01, -5.0])
    splitter = BufferedLOO(buffer_km=30.0)
    train_idx, test_idx = next(splitter.split(lats, lons))
    assert test_idx[0] == 0
    assert 1 not in train_idx


def test_recommend_block_size() -> None:
    assert recommend_block_size_km(40.0) >= 60.0


def test_variogram_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.normal(size=(n, 2))
    preds = rng.normal(size=n)
    res = preds * 0.1 + rng.normal(scale=0.05, size=n)
    out = compute_residual_variogram(preds, res, coords)
    assert out["range_km"] > 0


def test_spatial_holdout_fraction() -> None:
    lats = np.linspace(5, 8, 300)
    lons = np.linspace(-6, -3, 300)
    mask = spatial_holdout_mask(lats, lons, fraction=0.1, seed=1)
    assert 0.03 <= mask.mean() <= 0.35


def test_optimised_random_deterministic() -> None:
    lats = np.linspace(5, 7, 100)
    lons = np.linspace(-5, -3, 100)
    s1 = SpatialBlockSplit(strategy="optimised_random", seed=99, n_candidates=20)
    s2 = SpatialBlockSplit(strategy="optimised_random", seed=99, n_candidates=20)
    _, t1 = next(s1.split(lats, lons))
    _, t2 = next(s2.split(lats, lons))
    assert np.array_equal(t1, t2)
