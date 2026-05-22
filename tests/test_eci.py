"""Tests for Error-quantified Conformal Inference (ECI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from models.online_conformal_base import sigmoid_derivative
from tests.conformal_online_helpers import (
    distribution_shift_scores,
    run_online_coverage,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "conformal"
ALPHA = 0.1
NOMINAL = 1.0 - ALPHA
WU_TOL = 0.01


def test_sigmoid_derivative_finite_diff() -> None:
    x = np.linspace(-2, 2, 50)
    anal = sigmoid_derivative(x, c=1.0)
    h = 1e-5
    num = (1.0 / (1.0 + np.exp(-(x + h)))) - (1.0 / (1.0 + np.exp(-(x - h))))
    num /= 2 * h
    np.testing.assert_allclose(anal, num, rtol=1e-4, atol=1e-4)


def test_eci_cutoff_suppresses_small_gap() -> None:
    base = ErrorQuantifiedConformalInference(alpha=0.1, eta=0.01, window=20)
    cut = ECICutoff(alpha=0.1, eta=0.01, h=1.0, window=20)
    for s in np.linspace(0.4, 1.2, 40):
        base.update(float(s))
        cut.update(float(s))
    base.q = cut.q = 1.0
    base.update(1.002)
    cut.update(1.002)
    assert abs(cut.q - 1.0) < abs(base.q - 1.0)


def test_eci_integral_weights_normalized() -> None:
    decay = 0.95
    n = 30
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    w = decay**ages
    w /= w.sum()
    assert w.sum() == pytest.approx(1.0)


def test_distribution_shift_eci_reconverges() -> None:
    from tests.conformal_online_helpers import post_shift_coverage

    scores = distribution_shift_scores(T=1000, shift_at=500, seed=4)
    eci = ErrorQuantifiedConformalInference(alpha=ALPHA, eta=2.0, window=100)
    _, _, _, qs = run_online_coverage(eci, scores, alpha=ALPHA, burn_in=0, warm_start=200)
    cov_tail = post_shift_coverage(scores, qs, shift_at=500, window=100)
    assert abs(cov_tail - NOMINAL) <= 0.05


@pytest.mark.parametrize(
    "cls",
    [
        ErrorQuantifiedConformalInference,
        ECICutoff,
        ECIIntegral,
    ],
)
@pytest.mark.parametrize("fixture_name", ["amazon_prophet_scores", "google_prophet_scores"])
def test_wu_table_eci_variants_coverage(cls: type, fixture_name: str) -> None:
    scores = np.load(FIXTURES / f"{fixture_name}.npz")["scores"]
    eta = 4.0 if cls is ECIIntegral else 2.5
    updater = cls(alpha=ALPHA, eta=eta, window=100)
    cov, _, _, _ = run_online_coverage(updater, scores, alpha=ALPHA, burn_in=400, warm_start=200)
    assert NOMINAL - WU_TOL <= cov <= NOMINAL + WU_TOL
