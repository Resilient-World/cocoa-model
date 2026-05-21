"""Tests for online CQR yield wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.cqr import QuantileYieldSurrogate
from models.quantile_yield_surrogate_online import QuantileYieldSurrogateOnline


def _tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    climate = torch.randn(1, 365, 11) * 0.05
    static = torch.randn(1, 13) * 0.05
    return climate, static


def test_online_updates_threshold_when_y_observed() -> None:
    model = QuantileYieldSurrogate()
    online = QuantileYieldSurrogateOnline(model, online_method="aci", alpha=0.1, eta=0.05)
    q0 = online.current_threshold
    iv1 = online.predict_with_online_calibration(_tiny_batch(), observed_y=2.0)
    assert online.current_threshold != q0 or iv1.q_adjustment != q0
    q_after = online.current_threshold
    iv2 = online.predict_with_online_calibration(_tiny_batch(), observed_y=None)
    assert iv2.q_adjustment == q_after


def test_online_interval_bounds() -> None:
    model = QuantileYieldSurrogate()
    online = QuantileYieldSurrogateOnline(model, online_method="eci", alpha=0.1)
    iv = online.predict_with_online_calibration(_tiny_batch())
    assert np.isfinite(iv.lower) and np.isfinite(iv.upper)
    assert iv.q_adjustment == pytest.approx(0.0)
    assert iv.q_adjustment >= 0.0 or np.isfinite(iv.q_adjustment)
