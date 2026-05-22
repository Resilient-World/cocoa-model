"""Tests for Conformal PID online calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.conformal_pid import ConformalPID
from models.online_conformal_base import adaptive_learning_rate
from tests.conformal_online_helpers import (
    distribution_shift_scores,
    run_online_coverage,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "conformal"
ALPHA = 0.1
NOMINAL = 1.0 - ALPHA
WU_TOL = 0.01


def test_pid_integrator_bounded() -> None:
    pid = ConformalPID(alpha=0.1, eta=0.05, K_I=2.0, saturation_factor=0.05)
    for _ in range(200):
        pid.update(5.0)
    assert abs(pid._integrator()) <= pid.K_I + 1e-6


def test_adaptive_eta_scales_with_score_range() -> None:
    from collections import deque

    window = deque([0.0, 0.1, 0.9, 1.0], maxlen=100)
    eta_t = adaptive_learning_rate(window, 0.1, window=100)
    assert eta_t == pytest.approx(0.1 * 1.0)


def test_distribution_shift_pid_reconverges() -> None:
    from tests.conformal_online_helpers import post_shift_coverage

    scores = distribution_shift_scores(T=1000, shift_at=500, seed=3)
    pid = ConformalPID(alpha=ALPHA, eta=0.05, window=100)
    _, _, _, qs = run_online_coverage(pid, scores, alpha=ALPHA, burn_in=0)
    cov_tail = post_shift_coverage(scores, qs, shift_at=500, window=100)
    assert abs(cov_tail - NOMINAL) <= 0.05


@pytest.mark.parametrize("fixture_name", ["amazon_prophet_scores", "google_prophet_scores"])
def test_wu_table_pid_coverage(fixture_name: str) -> None:
    scores = np.load(FIXTURES / f"{fixture_name}.npz")["scores"]
    pid = ConformalPID(alpha=ALPHA, eta=0.01, window=100)
    cov, _, _, _ = run_online_coverage(pid, scores, alpha=ALPHA, burn_in=200, warm_start=200)
    assert NOMINAL - WU_TOL <= cov <= NOMINAL + WU_TOL
