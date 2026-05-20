"""Tests for conformalized quantile regression (CQR) yield uncertainty."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.cqr import (
    ConformalCalibrator,
    QuantileYieldSurrogate,
    pinball_loss,
)


def _synthetic_regression_data(
    n: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    climate = torch.randn(n, 365, 11) * 0.05
    climate[..., 0] = 28 + torch.randn(n, 365) * 0.5
    climate[..., 1] = 22 + torch.randn(n, 365) * 0.5
    climate[..., 2] = 25 + torch.randn(n, 365) * 0.5
    climate[..., 3] = torch.relu(torch.randn(n, 365)) * 2
    climate[..., 4] = 15.0
    climate[..., 5] = 1.0
    climate[..., 6] = 3.5
    climate[..., 7] = 0.3
    climate[..., 8] = 2.0
    climate[..., 9] = 75.0
    climate[..., 10] = 415.0
    static = torch.randn(n, 13) * 0.05
    static[:, 0] = 150.0
    latent = static[:, 0] * 0.004 + climate[..., 2].mean(dim=1) * 0.05
    y = latent + torch.randn(n) * 0.12
    y = y.clamp(min=0.3)
    return climate, static, y


def test_pinball_loss_nonnegative() -> None:
    pred = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    target = torch.tensor([2.0, 2.0])
    loss = pinball_loss(pred, target)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_quantile_forward_shape() -> None:
    model = QuantileYieldSurrogate()
    c, s, y = _synthetic_regression_data(4, seed=0)
    out = model(c, s)
    assert out.shape == (4, 3)


def test_conformal_calibrator_perfect_quantiles_cover() -> None:
    """Ideal q05/q95 plus conformal adjustment should reach ~80% on test."""
    rng = np.random.default_rng(7)
    n_cal, n_test = 400, 400
    y_cal = rng.normal(2.0, 0.25, n_cal)
    q_lo_cal = y_cal - 0.35
    q_hi_cal = y_cal + 0.35

    calibrator = ConformalCalibrator()
    scores = calibrator.conformity_scores(y_cal, q_lo_cal, q_hi_cal)
    calibrator.alpha = 0.2
    calibrator.Q_hat = calibrator._conformal_quantile(scores, alpha=0.2)

    y_test = rng.normal(2.0, 0.25, n_test)
    q_lo_test = y_test - 0.35
    q_hi_test = y_test + 0.35
    lowers = q_lo_test - calibrator.Q_hat
    uppers = q_hi_test + calibrator.Q_hat
    coverage = calibrator.empirical_coverage_on(y_test, lowers, uppers)
    assert coverage >= 0.78


def test_cqr_end_to_end_coverage_on_synthetic_panel() -> None:
    """
    Train small CQR model + calibrator; held-out coverage >= 78% at 80% nominal.

    Reference: HSE-GNN-CP crop-yield CQR validation (MDPI Information 2024, 17(2):141).
    """
    torch.manual_seed(0)
    climate, static, y = _synthetic_regression_data(900, seed=1)
    n = climate.shape[0]
    idx = torch.randperm(n)
    n_train = int(0.7 * n)
    n_cal = int(0.15 * n)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train : n_train + n_cal]
    test_idx = idx[n_train + n_cal :]

    model = QuantileYieldSurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(25):
        model.train()
        pred = model(climate[train_idx], static[train_idx])
        loss = pinball_loss(pred, y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    calibrator = ConformalCalibrator().fit(
        model,
        (climate[cal_idx], static[cal_idx]),
        y[cal_idx],
        alpha=0.2,
    )
    lowers, _, uppers = calibrator.predict_interval_batch(
        model,
        (climate[test_idx], static[test_idx]),
    )
    coverage = calibrator.empirical_coverage_on(
        y[test_idx].numpy(),
        lowers,
        uppers,
    )
    assert coverage >= 0.78


def test_calibrator_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "cal.joblib"
    cal = ConformalCalibrator()
    cal.alpha = 0.2
    cal.Q_hat = 0.15
    cal.empirical_coverage = 0.81
    cal.save(path)
    loaded = ConformalCalibrator.load(path)
    assert loaded.Q_hat == pytest.approx(0.15)
    assert loaded.alpha == pytest.approx(0.2)
