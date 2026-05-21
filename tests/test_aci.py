"""Tests for Adaptive Conformal Inference (ACI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.aci import AdaptiveConformalInference, MultiStepACI, default_multistep_aci
from models.cqr import ConformalCalibrator
from tests.conformal_online_helpers import (
    distribution_shift_scores,
    run_online_coverage,
    split_cqr_static_coverage,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "conformal"
ALPHA = 0.1
NOMINAL = 1.0 - ALPHA
WU_TOL = 0.01


def test_aci_update_matches_manual_step() -> None:
    aci = AdaptiveConformalInference(alpha=0.1, eta=0.05, q_init=1.0)
    score = 1.5
    q_before = aci.q
    err = 1.0 if score > q_before else 0.0
    expected = q_before + 0.05 * (err - 0.1)
    assert aci.update(score) == pytest.approx(expected)


def test_gibbs_candes_finite_sample_bound() -> None:
    alpha = 0.1
    eta = 0.02
    eps_1 = 0.05
    T = 500
    rng = np.random.default_rng(0)
    aci = AdaptiveConformalInference(alpha=alpha, eta=eta, eps_1=eps_1)
    scores = rng.exponential(0.5, T)
    cov, _, _, _ = run_online_coverage(aci, scores, alpha=alpha, burn_in=100)
    bound = aci.finite_sample_bound_rhs(T)
    assert abs(cov - NOMINAL) <= bound + 0.05


def test_distribution_shift_aci_reconverges() -> None:
    scores = distribution_shift_scores(T=1000, shift_at=500, seed=1)
    aci = AdaptiveConformalInference(alpha=ALPHA, eta=0.05, q_init=0.0)
    _, _, _, qs = run_online_coverage(aci, scores, alpha=ALPHA, burn_in=0)
    from tests.conformal_online_helpers import post_shift_coverage

    cov_tail = post_shift_coverage(scores, qs, shift_at=500, window=100)
    assert abs(cov_tail - NOMINAL) <= 0.05


def test_split_cqr_fails_under_shift() -> None:
    scores = distribution_shift_scores(T=1000, shift_at=500, seed=2)
    static_cov = split_cqr_static_coverage(scores, alpha=ALPHA, cal_fraction=0.5)
    assert static_cov < 0.85


def test_multistep_aci_equal_horizon_coverage() -> None:
    rng = np.random.default_rng(10)
    horizons = ["2030", "2050", "2080"]
    msa = default_multistep_aci(alpha=ALPHA, eta=0.05, horizons=horizons)
    for _ in range(500):
        score_vec = rng.exponential(0.4, len(horizons))
        msa.update(score_vec, None)
    coverages = list(msa.empirical_coverage_by_horizon().values())
    assert max(coverages) - min(coverages) <= 0.03


@pytest.mark.parametrize("fixture_name", ["amazon_prophet_scores", "google_prophet_scores"])
def test_wu_table2_coverage_band(fixture_name: str) -> None:
    path = FIXTURES / f"{fixture_name}.npz"
    data = np.load(path)
    scores = data["scores"]
    aci = AdaptiveConformalInference(alpha=ALPHA, eta=0.03, q_init=0.0)
    cov, _, _, _ = run_online_coverage(
        aci, scores, alpha=ALPHA, burn_in=200, warm_start=200
    )
    assert NOMINAL - WU_TOL <= cov <= NOMINAL + WU_TOL
