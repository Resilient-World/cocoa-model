"""Avoided-loss intervention simulation using the yield surrogate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor

from api.financial import calculate_financial_impact_usd
from api.feature_resolver import climate_tensor_from_dataset_point
from api.schemas import (
    AvoidedLossInterval,
    AvoidedLossUncertaintyBand,
    ConfidenceInterval,
    ConformalConfidenceInterval,
    ConformalIntervalResponse,
    InterventionType,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
    SimulateScenarioRequest,
    SimulateScenarioResponse,
    YieldUncertaintyBand,
)
from counterfactual.cmip6_scenarios import ScenarioBuilder
from hazards import apply_biotic_losses
from hazards.black_pod import ShadeSpecies
from models.yield_surrogate import CLIMATE_IDX, YieldSurrogateModel

if TYPE_CHECKING:
    from api.feature_resolver import FarmFeatureResolver
    from models.conformal import ConformalPredictor

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
        "shade_species": ShadeSpecies.KHAYA_IVORENSIS.value,
    },
    InterventionType.agroforestry: {
        "tmax": -1.0,
        "vpd_mult": 0.90,
        "sm_root": 0.05,
        "shade_species": ShadeSpecies.KHAYA_IVORENSIS.value,
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


def _climate_tensor_to_dataset(climate: Tensor, year: int) -> xr.Dataset:
    """Convert resolved daily climate tensor ``[1, T, 11]`` to an ``xr.Dataset`` for hazards."""
    arr = climate.squeeze(0).detach().cpu().numpy()
    n_days = arr.shape[0]
    time = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    data_vars = {
        name: ("time", arr[:, idx].astype(np.float32))
        for name, idx in CLIMATE_IDX.items()
    }
    return xr.Dataset(data_vars, coords={"time": time})


def _biotic_static_features(
    intervention_type: InterventionType | None,
    *,
    cssvd_prevalence_pct: float = 15.0,
    cssvd_tolerance: float = 1.0,
) -> dict[str, Any]:
    """Farm static covariates for biotic loss (CRIG prevalence mock until raster wired)."""
    shade = ShadeSpecies.UNSHADED
    if intervention_type is not None:
        deltas = INTERVENTION_CLIMATE_DELTAS.get(intervention_type, {})
        raw_shade = deltas.get("shade_species")
        if raw_shade is not None:
            shade = ShadeSpecies(str(raw_shade))
    return {
        "cssvd_prevalence_pct": cssvd_prevalence_pct,
        "cssvd_tolerance": cssvd_tolerance,
        "shade_species": shade,
    }


def _biotic_response_block(result: dict[str, Any]) -> dict[str, Any]:
    from api.schemas import BioticLossAttribution, ScenarioBioticLosses

    attr = result["loss_attribution"]
    return ScenarioBioticLosses(
        surviving_fraction=result["surviving_fraction"],
        total_loss_fraction=result["total_loss_fraction"],
        loss_attribution=BioticLossAttribution(
            black_pod=attr["black_pod"],
            cssvd=attr["cssvd"],
            mirids=attr["mirids"],
        ),
    )


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


def _conformal_to_response(interval: Any) -> ConformalIntervalResponse:
    return ConformalIntervalResponse(
        point=interval.point,
        lower=interval.lower,
        upper=interval.upper,
        coverage_target=interval.coverage_target,
        method=interval.method,
        coverage_guarantee=interval.coverage_guarantee,
    )


def _conformal_avoided_loss_interval(
    baseline: ConformalIntervalResponse,
    projected: ConformalIntervalResponse,
    farm_size_ha: float,
) -> ConformalIntervalResponse:
    """Conservative conformal bounds on avoided loss from per-scenario yield intervals."""
    avoided_lower = max(0.0, (projected.lower - baseline.upper) * farm_size_ha)
    avoided_upper = max(0.0, (projected.upper - baseline.lower) * farm_size_ha)
    point = max(0.0, (projected.point - baseline.point) * farm_size_ha)
    return ConformalIntervalResponse(
        point=point,
        lower=avoided_lower,
        upper=avoided_upper,
        coverage_target=baseline.coverage_target,
        method=f"derived:{baseline.method}",
        coverage_guarantee=baseline.coverage_guarantee,
    )


def _blend_yield(mc_mean: float, current_yield: float, blend_weight: float) -> float:
    """Blend model output with observed yield for stable demo responses."""
    w = min(max(blend_weight, 0.0), 1.0)
    return (1.0 - w) * mc_mean + w * current_yield


def _blend_mc_numpy(samples: np.ndarray, current_yield: float, blend_weight: float) -> np.ndarray:
    """Blend each Monte Carlo draw toward the observed yield (demo stabilization)."""
    w = min(max(blend_weight, 0.0), 1.0)
    return (1.0 - w) * samples + w * float(current_yield)


def _mean_p10_p90(samples: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(samples)),
        float(np.percentile(samples, 10.0)),
        float(np.percentile(samples, 90.0)),
    )


def simulate_intervention(
    request: SimulateInterventionRequest,
    model: YieldSurrogateModel,
    feature_resolver: FarmFeatureResolver,
    *,
    num_samples: int = 50,
    yield_blend_weight: float = 0.3,
    climate_year: int | None = None,
    conformal: ConformalPredictor | None = None,
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

    ds_cf = _climate_tensor_to_dataset(climate_cf, year)
    ds_factual = _climate_tensor_to_dataset(climate_factual, year)
    biotic_cf = apply_biotic_losses(
        1.0,
        ds_cf,
        _biotic_static_features(None),
    )
    biotic_factual = apply_biotic_losses(
        1.0,
        ds_factual,
        _biotic_static_features(request.intervention_type),
    )
    samples_cf = samples_cf * biotic_cf["surviving_fraction"]
    samples_factual = samples_factual * biotic_factual["surviving_fraction"]

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

    conformal_block: ConformalConfidenceInterval | None = None
    if conformal is not None:
        from models.conformal import MondrianConformalYield

        predict_kwargs: dict[str, Any] = {
            "num_samples": num_samples,
            "device": "cpu",
        }
        if isinstance(conformal, MondrianConformalYield):
            predict_kwargs["lat"] = lat
            predict_kwargs["lon"] = lon

        cf_interval = conformal.predict(model, climate_cf, static_cf, **predict_kwargs)
        factual_interval = conformal.predict(
            model, climate_factual, static_factual, **predict_kwargs
        )
        baseline_cf = _conformal_to_response(cf_interval)
        projected_cf = _conformal_to_response(factual_interval)
        conformal_block = ConformalConfidenceInterval(
            baseline_yield_tonnes_per_ha=baseline_cf,
            projected_yield_tonnes_per_ha=projected_cf,
            avoided_loss_tonnes=_conformal_avoided_loss_interval(
                baseline_cf,
                projected_cf,
                request.farm_size_ha,
            ),
        )

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
        conformal_interval=conformal_block,
        biotic_loss_attribution={
            "baseline": _biotic_response_block(biotic_cf),
            "projected": _biotic_response_block(biotic_factual),
        },
    )


@torch.no_grad()
def simulate_scenario(
    request: SimulateScenarioRequest,
    model: YieldSurrogateModel,
    feature_resolver: FarmFeatureResolver,
    *,
    historical_zarr_path: Path,
    cmip6_zarr_path: Path,
    num_samples: int = 50,
    yield_blend_weight: float = 0.3,
    climate_year: int | None = None,
) -> SimulateScenarioResponse:
    """
    Future-climate avoided loss using CMIP6 delta-change on ERA5 (`ScenarioBuilder`) +
    paired Monte Carlo forwards for mean / p10 / p90 bands.

    Baseline is SSP-conditioned climate **without** intervention encoding; projected applies
    the usual mechanistic intervention deltas on the adjusted climate tensor.
    """
    if yield_blend_weight > 0.0:
        logger.warning(
            "yield_blend_weight=%.2f is a demo crutch; set to 0.0 once a trained "
            "checkpoint is loaded.",
            yield_blend_weight,
        )

    if not historical_zarr_path.is_dir():
        raise ValueError(
            f"Historical ERA5 Zarr not found at {historical_zarr_path}. "
            "Export ERA5-Land to that path or set ERA5_ZARR_PATH."
        )
    if not cmip6_zarr_path.is_dir():
        raise ValueError(
            f"CMIP6 Zarr store not found at {cmip6_zarr_path}. "
            "Build an ensemble Zarr or set CMIP6_ZARR_PATH."
        )

    lat = request.farm_location.lat
    lon = request.farm_location.lon
    year = int(climate_year or 2023)
    horizon = int(request.horizon_year)
    window = (f"{horizon}-01-01", f"{horizon}-12-31")

    builder = ScenarioBuilder(
        str(historical_zarr_path.resolve()),
        str(cmip6_zarr_path.resolve()),
    )
    ds_scenario = builder.build_scenario(request.scenario, window)
    climate_scenario = climate_tensor_from_dataset_point(ds_scenario, lat, lon, year)

    static_base = feature_resolver.resolve_static_with_galileo(lat, lon, year)
    static_cf = _encode_static(
        static_base,
        current_yield=request.current_yield,
        intervention_type=None,
    )
    static_factual = _encode_static(
        static_base,
        current_yield=request.current_yield,
        intervention_type=request.intervention_type,
    )

    climate_baseline = climate_scenario
    climate_projected = _apply_intervention_climate(climate_scenario, request.intervention_type)

    samples_cf = predict_yield_samples(model, climate_baseline, static_cf, num_samples)
    samples_factual = predict_yield_samples(model, climate_projected, static_factual, num_samples)

    baseline_np = samples_cf.detach().cpu().numpy().reshape(-1)
    projected_np = samples_factual.detach().cpu().numpy().reshape(-1)

    baseline_blended = _blend_mc_numpy(baseline_np, request.current_yield, yield_blend_weight)
    projected_blended = _blend_mc_numpy(projected_np, request.current_yield, yield_blend_weight)

    b_mean, b_p10, b_p90 = _mean_p10_p90(baseline_blended)
    p_mean, p_p10, p_p90 = _mean_p10_p90(projected_blended)

    avoided_arr = np.maximum(projected_blended - baseline_blended, 0.0) * request.farm_size_ha
    a_mean, a_p10, a_p90 = _mean_p10_p90(avoided_arr)

    financial = calculate_financial_impact_usd(a_mean, request.cocoa_price_usd)

    return SimulateScenarioResponse(
        scenario=request.scenario,
        horizon_year=request.horizon_year,
        climate_reference_year=year,
        baseline_yield_tonnes_per_ha=YieldUncertaintyBand(mean=b_mean, p10=b_p10, p90=b_p90),
        projected_yield_tonnes_per_ha=YieldUncertaintyBand(mean=p_mean, p10=p_p10, p90=p_p90),
        avoided_loss_tonnes=AvoidedLossUncertaintyBand(mean=a_mean, p10=a_p10, p90=a_p90),
        financial_impact_usd_mean=financial,
    )
