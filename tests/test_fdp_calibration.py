"""Unit tests for FDP threshold calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_fdp_threshold import threshold_sweep_metrics


def test_threshold_sweep_perfect_f1_at_model_card_threshold() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.97, 0.98, 0.99, 0.1, 0.2, 0.05])
    thresholds = np.arange(0.5, 1.0, 0.01)
    metrics = threshold_sweep_metrics(y_true, y_prob, thresholds)
    row = metrics.loc[(metrics["threshold"] - 0.96).abs().idxmin()]
    assert row["f1"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
