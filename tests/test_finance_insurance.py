"""Tests for finance.insurance."""

from __future__ import annotations

import numpy as np
import pytest

from finance.insurance import aggregate_portfolio_var, parametric_payout, reinsurance_layer


def test_parametric_payout_full_shortfall() -> None:
    payout = parametric_payout(0.5, 1.0, 10_000.0, basis_risk_factor=0.15)
    assert payout == pytest.approx(10_000.0 * 0.5 * 0.85)


def test_parametric_payout_no_shortfall() -> None:
    assert parametric_payout(1.2, 1.0, 10_000.0) == 0.0


def test_portfolio_var_increases_with_correlation() -> None:
    payouts = np.array([100.0, 120.0, 80.0, 110.0])
    low_rho = aggregate_portfolio_var(payouts, rho=0.1, alpha=0.99)
    high_rho = aggregate_portfolio_var(payouts, rho=0.8, alpha=0.99)
    assert high_rho > low_rho


def test_reinsurance_layer_cedes_excess() -> None:
    out = reinsurance_layer(500_000.0, attachment=100_000.0, exhaust=400_000.0)
    assert out["ceded"] == pytest.approx(300_000.0)
    assert out["net"] == pytest.approx(200_000.0)
