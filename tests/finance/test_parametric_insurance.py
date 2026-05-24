from __future__ import annotations

import numpy as np
import pytest

from finance.parametric_insurance import (
    compute_basis_risk,
    price_parametric_trigger,
    smile_corrected_pricing,
)


def test_basis_risk_zero_case() -> None:
    realized = np.array([0.0, 0.2, 0.4, 0.8])
    report = compute_basis_risk(realized, realized)
    assert report.rmse == pytest.approx(0.0)
    assert report.correlation == pytest.approx(1.0)
    assert report.regression_slope == pytest.approx(1.0)
    assert report.r2 == pytest.approx(1.0)


def test_basis_risk_one_mismatch_case() -> None:
    realized = np.array([0.0, 1.0, 0.0, 1.0])
    payout = 1.0 - realized
    report = compute_basis_risk(realized, payout)
    assert report.correlation == pytest.approx(-1.0)
    assert report.r2 == pytest.approx(0.0)
    assert report.rmse == pytest.approx(1.0)


def test_price_parametric_trigger_closed_form_expected_payout() -> None:
    samples = np.array([0.8, 1.0, 1.2])
    report = price_parametric_trigger(
        1.0,
        samples,
        {"lower": 0.9, "upper": 1.1},
        farm_size_ha=10.0,
        price_usd_per_t=3000.0,
    )
    assert report.expected_payout_usd == pytest.approx((0.2 * 10.0 * 3000.0) / 3.0)
    assert report.loaded_premium_usd >= report.fair_premium_usd
    assert 0.0 <= report.basis_risk_r2 <= 1.0


def test_smile_corrected_pricing() -> None:
    vols = smile_corrected_pricing([0.8, 1.0, 1.2], 0.2, {0.8: 1.3, 1.0: 1.0, 1.2: 1.4})
    assert vols.tolist() == pytest.approx([0.26, 0.2, 0.28])
