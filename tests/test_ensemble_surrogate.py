"""Tests for PINN ensemble surrogate (CocoaYieldPINN + YieldEnsemble)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from models.ensemble_surrogate import (
    N_CLIMATE,
    N_STATIC,
    SEQ_LEN,
    CocoaYieldPINN,
    YieldEnsemble,
    physics_residual_loss,
)


def _batch(n: int = 4) -> dict[str, torch.Tensor]:
    return {
        "X_climate": torch.randn(n, N_CLIMATE, SEQ_LEN),
        "X_static": torch.randn(n, N_STATIC),
        "y_case2": torch.rand(n).mul(1500),
        "y_almanac": torch.rand(n).mul(1500),
    }


def test_pinn_forward_shapes() -> None:
    model = CocoaYieldPINN()
    x_c = torch.randn(6, N_CLIMATE, SEQ_LEN)
    x_s = torch.randn(6, N_STATIC)
    out = model(x_c, x_s)
    assert out.shape == (6, 2)


def test_physics_residual_penalizes_yield_rise_under_drought() -> None:
    """Higher loss when yield increases with cwd (drier) vs decreases."""
    n, t = 8, SEQ_LEN
    cwd = torch.linspace(0.0, 1.0, t).unsqueeze(0).expand(n, -1)
    climate = torch.zeros(n, N_CLIMATE, t, requires_grad=True)
    climate.data[:, 5, :] = cwd

    y_bad = climate[:, 5, :].mean(dim=1, keepdim=True).expand(-1, 2)
    y_good = (-climate[:, 5, :].mean(dim=1, keepdim=True)).expand(-1, 2)

    loss_bad = physics_residual_loss(y_bad, climate, lambda_phys=1.0).item()
    loss_good = physics_residual_loss(y_good, climate, lambda_phys=1.0).item()
    assert loss_bad > loss_good


def test_stacking_weights_sum_to_one_per_ecozone() -> None:
    df = pd.DataFrame(
        {
            "ecozone": ["A", "A", "A", "B", "B", "B"],
            "y_true": [800, 900, 1000, 600, 700, 800],
            "pinn_case2": [750, 850, 950, 580, 680, 780],
            "pinn_almanac": [820, 920, 1020, 610, 710, 810],
        }
    )
    ens = YieldEnsemble()
    ens.fit_stacking(df)
    for eco in ("A", "B"):
        w = ens.stacking_weights(eco)
        assert w.shape == (2,)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= 0).all()


def test_mc_dropout_returns_nonzero_std() -> None:
    torch.manual_seed(0)
    model = CocoaYieldPINN(dropout=0.3)
    x_c = torch.randn(5, N_CLIMATE, SEQ_LEN)
    x_s = torch.randn(5, N_STATIC)

    samples = model.mc_predict_heads(x_c, x_s, n_samples=30)
    std = samples.std(dim=0).mean()
    assert std.item() > 0.0


def test_yield_ensemble_predict_uncertainty() -> None:
    torch.manual_seed(1)
    models = [CocoaYieldPINN(dropout=0.3) for _ in range(2)]
    ens = YieldEnsemble(models=models, n_mc_samples=30)

    x_c = torch.randn(3, N_CLIMATE, SEQ_LEN)
    x_s = torch.randn(3, N_STATIC)
    pred = ens.predict({"X_climate": x_c, "X_static": x_s}, return_uncertainty=True)
    assert pred.mean.shape == (3,)
    assert pred.std.shape == (3,)
    assert (pred.std > 0).all()
    assert pred.p10.shape == (3,)
    assert pred.p90.shape == (3,)
