"""Avoided-loss intervention simulation using the yield surrogate."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from api.financial import calculate_financial_impact_usd
from api.schemas import (
    AvoidedLossInterval,
    ConfidenceInterval,
    InterventionType,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
)
from models.yield_surrogate import CLIMATE_IDX, YieldSurrogateModel

if TYPE_CHECKING:
    from api.feature_resolver import FarmFeatureResolver

logger = logging.getLogger(__name__)

# Static feature indices (must match feature_resolver + simulation encoding)
AWC_STATIC_IDX = 0
BASELINE_YIELD_STATIC_IDX = 2
INTERVENTION_STATIC_IDX = 3
STRESS_TOLERANCE_STATIC_IDX = 4

# Intervention uplift registry (deltas on resolved ERA5 features)
INTERVENTION_CLIMATE_DELTAS: dict[InterventionType, dict[str, float]] = {
    InterventionType.shade_trees: {
        "tmax": -1.5,
        "vpd_mult": 0.85,
        "sm_root": 0.03,
    },
    InterventionType.agroforestry: {
        "tmax": -1.0,
        "vpd_mult": 0.90,
        "sm_root": 0.05,
    },
    InterventionType.drought_resistant_variety: {
        "sm_root": 0.08,
    },
}

INTERVENTION_STATIC_DELTAS: dict[InterventionType, dict[str, float]] = {
    InterventionType.agroforestry: {"awc_mm": 20.0},
    InterventionType.drought_resistant_variety: {"stress_tolerance": 1.0},
}


def _encode_static(
    static: Tensor,
    *,
    current_yield: float,
    intervention_type: InterventionType | None,
) -> Tensor:
    """Inject observed yield and intervention-specific static encodings."""
    out = static.clone()
    out[0, BASELINE_YIELD_STATIC_IDX] = current_yield / 5.0
    if intervention_type is None:
        out[0, INTERVENTION_STATIC_IDX] = 0.0
    else:
        out[0, INTERVENTION_STATIC_IDX] = 1.0
        static_deltas = INTERVENTION_STATIC_DELTAS.get(intervention_type, {})
        if "awc_mm" in static_deltas:
            out[0, AWC_STATIC_IDX] = out[0, AWC_STATIC_IDX] + static_deltas["awc_mm"]
        if static_deltas.get("stress_tolerance"):
            out[0, STRESS_TOLERANCE_STATIC_IDX] = 1.0
    return out


def _apply_intervention_climate(
    climate: Tensor,
    intervention_type: InterventionType,
) -> Tensor:
    """Apply mechanistic microclimate adjustments on resolved daily features."""
    out = climate.clone()
    deltas = INTERVENTION_CLIMATE_DELTAS.get(intervention_type, {})

    if "tmax" in deltas:
        out[..., CLIMATE_IDX["tmax"]] = out[..., CLIMATE_IDX["tmax"]] + deltas["tmax"]
        out[..., CLIMATE_IDX["tmean"]] = 0.5 * (
            out[..., CLIMATE_IDX["tmax"]] + out[..., CLIMATE_IDX["tmin"]]
        )

    if "vpd_mult" in deltas:
        out[..., CLIMATE_IDX["vpd"]] = (
            out[..., CLIMATE_IDX["vpd"]] * deltas["vpd_mult"]
        ).clamp(min=0.05)

    if "sm_root" in deltas:
        out[..., CLIMATE_IDX["sm_root"]] = (
            out[..., CLIMATE_IDX["sm_root"]] + deltas["sm_root"]
        ).clamp(0.05, 0.55)

    return out


@torch.no_grad()
def predict_yield_samples(
    model: YieldSurrogateModel,
    climate: Tensor,
    static: Tensor,
    num_samples: int,
) -> Tensor:
    """Run stochastic forward passes; returns ``[num_samples]`` yields."""
    was_training = model.training
    model.eval()
    samples = torch.stack(
        [model(climate, static).squeeze(0) for _ in range(num_samples)],
        dim=0,
    )
    if was_training:
        model.train()
    return samples


def _blend_yield(mc_mean: float, current_yield: float, blend_weight: float) -> float:
    """Blend model output with observed yield for stable demo responses."""
    w = min(max(blend_weight, 0.0), 1.0)
    return (1.0 - w) * mc_mean + w * current_yield


def simulate_intervention(
    request: SimulateInterventionRequest,
    model: YieldSurrogateModel,
    feature_resolver: FarmFeatureResolver,
    *,
    num_samples: int = 50,
    yield_blend_weight: float = 0.3,
    climate_year: int | None = None,
) -> SimulateInterventionResponse:
    """
    Predict counterfactual vs factual yield and compute avoided loss + financial impact.

    Uses :class:`~api.feature_resolver.FarmFeatureResolver` for ERA5/static features and
    paired Monte Carlo samples for a 90% confidence interval on avoided loss.
    """
    if yield_blend_weight > 0.0:
        logger.warning(
            "yield_blend_weight=%.2f is a demo crutch; set to 0.0 once a trained "
            "checkpoint is loaded.",
            yield_blend_weight,
        )

    lat = request.farm_location.lat
    lon = request.farm_location.lon
    year = climate_year or 2023

    climate_base = feature_resolver.resolve_climate(lat, lon, year)
    static_base = feature_resolver.resolve_static_with_galileo(lat, lon, year)

    static_cf = _encode_static(static_base, current_yield=request.current_yield, intervention_type=None)
    static_factual = _encode_static(
        static_base,
        current_yield=request.current_yield,
        intervention_type=request.intervention_type,
    )
    climate_cf = climate_base
    climate_factual = _apply_intervention_climate(climate_base, request.intervention_type)

    samples_cf = predict_yield_samples(model, climate_cf, static_cf, num_samples)
    samples_factual = predict_yield_samples(model, climate_factual, static_factual, num_samples)

    mc_baseline = float(samples_cf.mean().item())
    mc_projected_raw = float(samples_factual.mean().item())

    baseline_yield = _blend_yield(mc_baseline, request.current_yield, yield_blend_weight)
    projected_yield = _blend_yield(mc_projected_raw, request.current_yield, yield_blend_weight)

    delta_per_ha_samples = (samples_factual - samples_cf).cpu().numpy()
    avoided_per_ha_samples = np.maximum(delta_per_ha_samples, 0.0)
    avoided_loss_samples = avoided_per_ha_samples * request.farm_size_ha

    avoided_loss_tonnes = max(0.0, (projected_yield - baseline_yield) * request.farm_size_ha)
    financial_impact_usd = calculate_financial_impact_usd(
        avoided_loss_tonnes,
        request.cocoa_price_usd,
    )

    ci_lower = float(np.percentile(avoided_loss_samples, 5.0))
    ci_upper = float(np.percentile(avoided_loss_samples, 95.0))

    return SimulateInterventionResponse(
        baseline_yield_tonnes_per_ha=baseline_yield,
        projected_yield_tonnes_per_ha=projected_yield,
        avoided_loss_tonnes=avoided_loss_tonnes,
        financial_impact_usd=financial_impact_usd,
        confidence_interval=ConfidenceInterval(
            avoided_loss_tonnes=AvoidedLossInterval(
                lower=ci_lower,
                upper=ci_upper,
                level=0.9,
            ),
        ),
    )
