"""Blocked CQR calibration smoke test."""

from __future__ import annotations

import numpy as np
import torch

from models.conformal.cqr import ConformalCalibrator, QuantileYieldSurrogate
from validation.conformal_cv import _synthetic_panel_rows, evaluate_cv_strategy


def test_spatial_block_vs_random_coverage_smoke() -> None:
    rows = _synthetic_panel_rows(120, seed=0)
    spatial = evaluate_cv_strategy("spatial_block", rows, block_size_km=80.0, seed=0)
    random = evaluate_cv_strategy("random", rows, seed=0)
    assert "coverage" in spatial
    assert "coverage" in random
    assert spatial.get("production_target") is True


def test_fit_blocked_runs() -> None:
    from validation.spatial_cv import SpatialBlockSplit

    n = 40
    climate = torch.randn(n, 365, 11)
    static = torch.randn(n, 13)
    y = torch.rand(n) + 0.5
    lats = np.linspace(5, 7, n)
    lons = np.linspace(-6, -3, n)
    model = QuantileYieldSurrogate()
    cal = ConformalCalibrator().fit_blocked(
        model,
        (climate, static),
        y,
        SpatialBlockSplit(block_size_km=100.0, n_folds=4, seed=0),
        coords=(lats, lons),
    )
    assert cal.Q_hat is not None
    assert cal.cv_strategy == "spatial_block"
