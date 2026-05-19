"""Avoided-loss intervention simulation using the yield surrogate."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from api.financial import calculate_financial_impact_usd
from api.geo_mock import fetch_climate_and_soil
from api.schemas import (
    AvoidedLossInterval,
    ConfidenceInterval,
    InterventionType,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
)
from models.yield_surrogate import CLIMATE_IDX, YieldSurrogateModel

logger = logging.getLogger(__name__)

# Static feature indices (must match geo_mock + simulation encoding)
AWC_STATIC_IDX = 0
BASELINE_YIELD_STATIC_IDX = 2
INTERVENTION_STATIC_IDX = 3
STRESS_TOLERANCE_STATIC_IDX = 4

# Lahive 2019 microclimate: typical shade effects (extreme cases up to ~7 °C / larger VPD cuts)
SHADE_TMAX_DELTA_C = -2.0
SHADE_VPD_DELTA_KPA = -1.0
AGROFORESTRY_VPD_DELTA_KPA = -0.5
AGROFORESTRY_AWC_DELTA_MM = 20.0


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
        if intervention_type == InterventionType.agroforestry:
            out[0, AWC_STATIC_IDX] = out[0, AWC_STATIC_IDX] + AGROFORESTRY_AWC_DELTA_MM
        elif intervention_type == InterventionType.drought_resistant_variety:
            out[0, STRESS_TOLERANCE_STATIC_IDX] = 1.0
    return out


def _apply_intervention_climate(
    climate: Tensor,
    intervention_type: InterventionType,
) -> Tensor:
    """Microclimate adjustments for intervention scenarios (mechanistic channels)."""
    out = climate.clone()
    if intervention_type == InterventionType.shade_trees:
        out[..., CLIMATE_IDX["tmax"]] = out[..., CLIMATE_IDX["tmax"]] + SHADE_TMAX_DELTA_C
        out[..., CLIMATE_IDX["vpd"]] = (out[..., CLIMATE_IDX["vpd"]] + SHADE_VPD_DELTA_KPA).clamp(
            min=0.05
        )
    elif intervention_type == InterventionType.agroforestry:
        out[..., CLIMATE_IDX["vpd"]] = (
            out[..., CLIMATE_IDX["vpd"]] + AGROFORESTRY_VPD_DELTA_KPA
        ).clamp(min=0.05)
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
    *,
    num_samples: int = 50,
    yield_blend_weight: float = 0.3,
) -> SimulateInterventionResponse:
    """
    Predict counterfactual vs factual yield and compute avoided loss + financial impact.

    Uses paired Monte Carlo samples for a 90% confidence interval on avoided loss.
    """
    if yield_blend_weight > 0.0:
        logger.warning(
            "yield_blend_weight=%.2f is a demo crutch; set to 0.0 once a trained "
            "checkpoint is loaded.",
            yield_blend_weight,
        )

    lat = request.farm_location.lat
    lon = request.farm_location.lon
    climate_base, static_base = fetch_climate_and_soil(lat, lon)

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
