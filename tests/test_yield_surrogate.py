"""Unit tests for PINN yield surrogate, mechanistic core, and uncertainty."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from models.checkpoint_migration import is_v1_static_checkpoint, migrate_v1_static_to_v2
from models.yield_surrogate import (
    CocoaPINNLoss,
    DeepEnsemble,
    MCDropout,
    N_STATIC_SITE,
    PhysicsInformedYieldLoss,
    STATIC_FEATURE_NAMES,
    YieldSurrogateModel,
    cohort_phase_from_age,
    pack_tree_age_static,
    predict_with_uncertainty,
)


def _dummy_batch(
    B: int = 4,
    T: int = 365,
    C: int = 11,
    S: int = N_STATIC_SITE,
) -> tuple[torch.Tensor, torch.Tensor]:
    climate = torch.randn(B, T, C) * 0.1
    climate[..., 0] = 30 + torch.randn(B, T) * 2  # tmax
    climate[..., 1] = 22 + torch.randn(B, T) * 1  # tmin
    climate[..., 2] = 26 + torch.randn(B, T) * 1  # tmean
    climate[..., 3] = torch.relu(torch.randn(B, T)) * 3  # precip
    climate[..., 4] = 18 + torch.randn(B, T) * 1  # srad MJ/m2/d
    climate[..., 5] = 0.8 + torch.relu(torch.randn(B, T)) * 0.3  # vpd
    climate[..., 6] = 3.5 + torch.randn(B, T) * 0.3  # ET0
    climate[..., 7] = 0.28 + torch.randn(B, T) * 0.02
    climate[..., 8] = 2.0 + torch.randn(B, T) * 0.3
    climate[..., 9] = 80 + torch.randn(B, T) * 3
    climate[..., 10] = 415.0
    static = torch.randn(B, S)
    static[:, 0] = 150.0  # AWC
    return climate, static


# --- Forward / legacy / MC (original + new) ---


def test_forward_shape() -> None:
    m = YieldSurrogateModel()
    c, s = _dummy_batch()
    y = m(c, s)
    assert y.shape == (4,)


def test_forward_output_shape_legacy_batch() -> None:
    """Original: batch 8 with legacy 4-channel climate."""
    model = YieldSurrogateModel()
    climate = torch.randn(8, 365, 4)
    static = torch.randn(8, N_STATIC_SITE)
    static[:, 0] = 150.0
    pred = model(climate, static)
    assert pred.shape == (8,)


def test_legacy_4channel_still_works_with_warning() -> None:
    m = YieldSurrogateModel()
    c = torch.randn(2, 365, 4)
    s = torch.randn(2, N_STATIC_SITE)
    s[:, 0] = 150.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = m(c, s)
    assert any("deprecat" in str(x.message).lower() for x in w)
    assert y.shape == (2,)


def test_mc_dropout_uncertainty_nontrivial() -> None:
    m = YieldSurrogateModel()
    c, s = _dummy_batch(B=8)
    pred = predict_with_uncertainty(m, c, s, num_samples=20)
    assert pred.std.mean().item() > 0.0


def test_predict_with_uncertainty_shapes() -> None:
    model = YieldSurrogateModel(dropout=0.2)
    climate = torch.randn(4, 365, 4)
    static = torch.randn(4, N_STATIC_SITE)
    static[:, 0] = 150.0
    result = predict_with_uncertainty(model, climate, static, num_samples=50)
    assert result.mean.shape == (4,)
    assert result.std.shape == (4,)


def test_predict_with_uncertainty_nonzero_std() -> None:
    model = YieldSurrogateModel(dropout=0.3)
    climate = torch.randn(8, 365, 4)
    static = torch.randn(8, N_STATIC_SITE)
    static[:, 0] = 150.0
    result = predict_with_uncertainty(model, climate, static, num_samples=50)
    assert (result.std > 0).all()


def test_predict_with_uncertainty_single_sample() -> None:
    model = YieldSurrogateModel()
    climate = torch.randn(2, 365, 4)
    static = torch.randn(2, N_STATIC_SITE)
    static[:, 0] = 150.0
    result = predict_with_uncertainty(model, climate, static, num_samples=1)
    assert result.std.shape == (2,)
    assert torch.all(result.std == 0)


def test_mc_dropout_active_when_model_eval() -> None:
    layer = MCDropout(p=0.5)
    layer.eval()
    x = torch.ones(100, 32)
    torch.manual_seed(0)
    out1 = layer(x)
    torch.manual_seed(0)
    out2 = layer(x)
    assert not torch.allclose(out1, x)
    assert torch.allclose(out1, out2)


# --- Mechanistic physics (8 new) ---


def test_mechanistic_core_water_balance_never_negative() -> None:
    m = YieldSurrogateModel()
    c, s = _dummy_batch()
    c[..., 3] = 0.0  # no precip
    c[..., 6] = 5.0  # high ET0
    _, traces = m.forward_with_traces(c, s)
    assert (traces["sw_trace"] >= -1e-5).all()


def test_mechanistic_biomass_is_monotone_nondecreasing() -> None:
    m = YieldSurrogateModel()
    c, s = _dummy_batch()
    _, traces = m.forward_with_traces(c, s)
    diffs = traces["biomass_trace"].diff(dim=1)
    assert (diffs >= -1e-5).all()


def test_high_vpd_reduces_yield() -> None:
    m = YieldSurrogateModel()
    c1, s = _dummy_batch()
    c2 = c1.clone()
    c2[..., 5] = c1[..., 5] + 1.5  # push VPD past 1.8 kPa threshold
    _, t1 = m.forward_with_traces(c1, s)
    _, t2 = m.forward_with_traces(c2, s)
    assert t2["y_mech"].mean() < t1["y_mech"].mean()


def test_co2_enhancement_increases_yield() -> None:
    m = YieldSurrogateModel()
    c1, s = _dummy_batch()
    c2 = c1.clone()
    c2[..., 10] = 700.0
    _, t1 = m.forward_with_traces(c1, s)
    _, t2 = m.forward_with_traces(c2, s)
    assert t2["y_mech"].mean() > t1["y_mech"].mean()


def test_supraoptimal_temperature_reduces_yield() -> None:
    m = YieldSurrogateModel()
    c1, s = _dummy_batch()
    c2 = c1.clone()
    c2[..., 0] = c1[..., 0] + 16  # supraoptimal / near-lethal heat
    c2[..., 2] = c1[..., 2] + 16
    _, t1 = m.forward_with_traces(c1, s)
    _, t2 = m.forward_with_traces(c2, s)
    assert t2["y_mech"].mean() < t1["y_mech"].mean()


def test_drought_reduces_yield() -> None:
    m = YieldSurrogateModel()
    c1, s = _dummy_batch()
    c2 = c1.clone()
    c2[..., 3] = 0.0
    _, t1 = m.forward_with_traces(c1, s)
    _, t2 = m.forward_with_traces(c2, s)
    assert t2["y_mech"].mean() < t1["y_mech"].mean()


def test_pinn_loss_components_sum_correctly() -> None:
    m = YieldSurrogateModel()
    c, s = _dummy_batch()
    y_true = torch.full((4,), 1.5)
    loss_fn = CocoaPINNLoss(y_max=4.0)
    y_pred, traces = m.forward_with_traces(c, s)
    out = loss_fn(y_pred, y_true, traces, return_components=True)
    assert out["loss"].item() >= out["mse"].item() - 1e-6
    assert out["penalty_water"].item() >= 0


def test_deep_ensemble_predict_returns_mean_and_std() -> None:
    ens = DeepEnsemble(n_members=3)
    c, s = _dummy_batch()
    pred = ens.predict(c, s, num_mc_samples=5)
    assert pred.mean.shape == (4,)
    assert (pred.std > 0).all()


# --- Validation / legacy PhysicsInformedYieldLoss ---


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
    static = torch.randn(4, N_STATIC_SITE)
    with pytest.raises(ValueError, match="sequence_length"):
        model(climate, static)


def test_invalid_static_features_raises() -> None:
    model = YieldSurrogateModel(static_features=N_STATIC_SITE)
    climate = torch.randn(4, 365, 4)
    static = torch.randn(4, 5)
    with pytest.raises(ValueError, match="static_features"):
        model(climate, static)


def test_static_feature_count_is_13() -> None:
    assert len(STATIC_FEATURE_NAMES) == 13
    assert YieldSurrogateModel().site_static_features == 13


def test_age_curve_peaks_at_12y() -> None:
    assert cohort_phase_from_age(12.0) == pytest.approx(1.0)
    assert cohort_phase_from_age(3.0) == pytest.approx(0.0)
    assert cohort_phase_from_age(7.0) == pytest.approx(0.5)
    assert cohort_phase_from_age(30.0) == pytest.approx(0.6)


def _static_with_tree_age(age_years: float) -> tuple[torch.Tensor, torch.Tensor]:
    climate = torch.randn(1, 365, 11)
    climate[..., 2] = 26.0
    climate[..., 4] = 15.0
    static = torch.zeros(1, N_STATIC_SITE)
    static[:, 0] = 150.0
    age_norm, cohort, dens = pack_tree_age_static(age_years)
    static[:, 10] = age_norm
    static[:, 11] = cohort
    static[:, 12] = dens
    return climate, static


def test_senescent_farm_yields_60pct_of_peak() -> None:
    model = YieldSurrogateModel()
    climate, s_peak = _static_with_tree_age(12.0)
    _, s_sen = _static_with_tree_age(30.0)
    traces_peak = model.mechanistic(climate, s_peak)
    traces_sen = model.mechanistic(climate, s_sen)
    ratio = traces_sen["y_mech"] / traces_peak["y_mech"]
    assert float(ratio.item()) == pytest.approx(0.6, rel=0.02)


def test_legacy_10dim_checkpoint_loads_via_migration() -> None:
    old_names = STATIC_FEATURE_NAMES[:10]
    model_v1 = YieldSurrogateModel(static_features=10, static_feature_names=old_names)
    state = model_v1.state_dict()
    assert is_v1_static_checkpoint(state)
    state_v2 = migrate_v1_static_to_v2(state)
    model_v2 = YieldSurrogateModel()
    model_v2.load_state_dict(state_v2, strict=False)
    climate, static = _static_with_tree_age(12.0)
    pred = model_v2(climate, static)
    assert pred.shape == (1,)
