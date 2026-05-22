"""Tests for probabilistic forecast scoring."""

from __future__ import annotations

import numpy as np
import pytest

from validation.calibration_metrics import run_calibration_gate
from validation.forecast_scoring import (
    crps_ensemble,
    crpss,
    energy_score,
    pit_histogram,
    reliability_diagram,
    sharpness,
)


def test_perfect_ensemble_near_zero_crps() -> None:
    obs = np.array([1.0, 2.0, 1.5])
    ens = np.stack([obs, obs, obs], axis=1)
    scores = crps_ensemble(obs, ens)
    assert np.all(scores < 0.05)


def test_crpss_manual() -> None:
    assert crpss(0.1, 0.2) == pytest.approx(0.5)


def test_reliability_perfect_quantiles() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    levels = np.array([0.05, 0.5, 0.95])
    qpred = np.column_stack(
        [
            np.quantile(y, 0.05) + np.zeros(500),
            np.quantile(y, 0.5) + np.zeros(500),
            np.quantile(y, 0.95) + np.zeros(500),
        ]
    )
    for j, tau in enumerate(levels):
        qpred[:, j] = np.quantile(y, tau)
    _, emp, ece, _ = reliability_diagram(y, qpred, levels)
    assert ece < 0.08


def test_pit_uniform_shape() -> None:
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 200)
    lo = np.zeros(200)
    hi = np.ones(200)
    _, _, _, diag = pit_histogram(y, lowers=lo, uppers=hi)
    assert diag["pit_chi2_p"] > 0.01 or diag["shape"] == "uniform"


def test_underdispersed_u_shape() -> None:
    y = np.linspace(0.05, 0.95, 100)
    lo = y * 0.35 + 0.325
    hi = y * 0.35 + 0.375
    _, _, _, diag = pit_histogram(y, lowers=lo, uppers=hi)
    assert diag["shape"] in ("u_shape", "skewed") or diag["pit_chi2_p"] < 0.01


def test_energy_score_perfect() -> None:
    obs = np.array([[1.0, 0.5], [2.0, 0.6]])
    ens = obs[:, None, :]
    scores = energy_score(obs, ens)
    assert np.all(scores < 1e-6)


def test_sharpness() -> None:
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 3.0])
    assert sharpness((lo, hi)) == pytest.approx(1.5)


def test_calibration_gate_fails_miscalibrated() -> None:
    report = {
        "nominal_coverage": 0.9,
        "empirical_coverage": 0.7,
        "pit_chi2_p": 0.001,
        "sharpness": 0.2,
        "pit_shape": "u_shape",
    }
    baseline = {"sharpness": 0.4, "empirical_coverage": 0.91, "pit_chi2_p": 0.2}
    ok, _ = run_calibration_gate(report, baseline)
    assert not ok


def test_calibration_gate_passes() -> None:
    report = {
        "nominal_coverage": 0.9,
        "empirical_coverage": 0.91,
        "pit_chi2_p": 0.2,
        "sharpness": 0.44,
    }
    baseline = {"sharpness": 0.45, "empirical_coverage": 0.9, "pit_chi2_p": 0.15}
    ok, msgs = run_calibration_gate(report, baseline)
    assert ok
    assert any("passed" in m for m in msgs)
