"""Unit tests for yield surrogate model and physics-informed loss."""

import pytest
import torch
import torch.nn.functional as F

from models.yield_surrogate import PhysicsInformedYieldLoss, YieldSurrogateModel


def test_forward_output_shape() -> None:
    model = YieldSurrogateModel()
    climate = torch.randn(8, 365, 4)
    static = torch.randn(8, 10)
    pred = model(climate, static)
    assert pred.shape == (8,)


def test_loss_no_penalty_when_below_ymax() -> None:
    y_max = 3.5
    loss_fn = PhysicsInformedYieldLoss(y_max=y_max, penalty_weight=100.0)
    pred = torch.tensor([1.0, 2.0, 3.0, 3.4])
    target = torch.tensor([1.1, 2.1, 2.9, 3.0])
    components = loss_fn(pred, target, return_components=True)
    assert isinstance(components, dict)
    assert components["penalty"].item() == pytest.approx(0.0, abs=1e-6)
    assert components["loss"].item() == pytest.approx(components["mse"].item(), rel=1e-5)


def test_loss_penalty_when_above_ymax() -> None:
    y_max = 3.5
    loss_fn = PhysicsInformedYieldLoss(y_max=y_max, penalty_weight=100.0)
    pred = torch.tensor([10.0, 10.0])
    target = torch.tensor([3.0, 3.0])
    mse_only = F.mse_loss(pred, target)
    total = loss_fn(pred, target)
    assert total.item() > mse_only.item()


def test_invalid_climate_shape_raises() -> None:
    model = YieldSurrogateModel(sequence_length=365, climate_features=4)
    climate = torch.randn(4, 100, 4)
    static = torch.randn(4, 10)
    with pytest.raises(ValueError, match="sequence_length"):
        model(climate, static)


def test_invalid_static_features_raises() -> None:
    model = YieldSurrogateModel(static_features=10)
    climate = torch.randn(4, 365, 4)
    static = torch.randn(4, 5)
    with pytest.raises(ValueError, match="static_features"):
        model(climate, static)
