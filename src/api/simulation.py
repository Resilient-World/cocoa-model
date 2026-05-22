"""Avoided-loss intervention simulation using the yield surrogate."""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor

from api.financial import calculate_financial_impact, financial_impact_to_schema
from api.feature_resolver import climate_tensor_from_dataset_point
from api.schemas import (
    AvoidedLossInterval,
    AvoidedLossUncertaintyBand,
    ConfidenceInterval,
    ConformalConfidenceInterval,
    ConformalIntervalResponse,
    InterventionType,
    SimulateClimateAttributionRequest,
    SimulateClimateAttributionResponse,
    SimulateInterventionRequest,
    SimulateInterventionResponse,
    SimulateScenarioRequest,
    SimulateScenarioResponse,
    YieldUncertaintyBand,
)
from counterfactual.cmip6_scenarios import ScenarioBuilder
from counterfactual.corrdiff_downscaler import (
    DEFAULT_OUTPUT_VARIABLES,
    CorrDiffCMIP6Downscaler,
    corrdiff_cache_missing_message,
    corrdiff_cache_path,
    load_corrdiff_scenario_ensemble,
)
from hazards import apply_biotic_losses
from hazards.black_pod import ShadeSpecies
from analysis.climate_attribution import extract_daily_climate_11ch
from models.casej_process import co2_ppm_for_ssp
from models.casej_surrogate import CASEJSurrogate
from api.scenario_conformal import apply_scenario_conformal, resolve_region
from models.cqr import ConformalCalibrator, QuantileYieldSurrogate
from models.yield_surrogate import CLIMATE_IDX, YieldSurrogateModel
from models.yield_surrogate_v2 import YieldSurrogateV2, region_id_from_country_code, region_id_from_latlon
from models.yield_surrogate_v2_teleconnection import YieldSurrogateV2Teleconnection

if TYPE_CHECKING:
    from api.feature_resolver import FarmFeatureResolver
    from api.schemas import UQMethod
    from models.conformal import ConformalPredictor
else:
    from api.schemas import UQMethod

log = structlog.get_logger(__name__)

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


def _dataset_to_case2_weather(ds: xr.Dataset) -> pd.DataFrame:
    """Map resolved climate dataset to CASE2/ALMANAC daily weather columns."""
    tmin = ds["tmin_c"].values.astype(float)
    tmax = ds["tmax_c"].values.astype(float)
    precip = ds["precip_mm"].values.astype(float)
    srad = ds.get("srad_mj", ds["precip_mm"] * 0 + 12.0).values.astype(float)
    vp = ds.get("vapor_pressure_kpa", ds["precip_mm"] * 0 + 1.5).values.astype(float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(ds["time"].values),
            "tmin_c": tmin,
            "tmax_c": tmax,
            "precip_mm": precip,
            "srad_mj": srad,
            "vapor_pressure_kpa": vp,
        }
    )


def _try_case2_yield_tonnes_ha(ds: xr.Dataset, *, n_years: int = 8) -> float | None:
    try:
        from models.process.case2_runner import CASE2NotInstalled, CASE2Runner
    except ImportError:
        return None
    try:
        runner = CASE2Runner()
        weather = _dataset_to_case2_weather(ds)
        result = runner.simulate(weather, soil={}, management={}, n_years=n_years)
        kg = float(np.nanmean(result.yearly_yield_kg_ha[-1:]))
        return kg / 1000.0
    except (CASE2NotInstalled, RuntimeError, ValueError, OSError) as exc:
        log.debug("CASE2 unavailable for process BMA: %s", exc)
        return None


def _try_almanac_yield_tonnes_ha(ds: xr.Dataset, *, n_years: int = 8) -> float | None:
    try:
        from models.process.almanac_runner import ALMANACNotInstalled, ALMANACRunner
    except ImportError:
        return None
    try:
        runner = ALMANACRunner()
        weather = _dataset_to_case2_weather(ds)
        result = runner.simulate(weather, soil={}, management={}, n_years=n_years)
        kg = float(np.nanmean(result.yearly_yield_kg_ha[-1:]))
        return kg / 1000.0
    except (ALMANACNotInstalled, RuntimeError, ValueError, OSError) as exc:
        log.debug("ALMANAC unavailable for process BMA: %s", exc)
        return None


def _apply_process_bma_to_means(
    *,
    request: SimulateScenarioRequest,
    settings: Any,
    model: CASEJSurrogate | YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    baseline_mean: float,
    projected_mean: float,
    climate_baseline: Tensor,
    climate_projected: Tensor,
    static_cf: Tensor,
    static_factual: Tensor,
    scenario_co2: float,
    year: int,
    lat: float,
    lon: float,
    country_code: str | None,
    feature_resolver: Any,
    horizon: int,
) -> tuple[float, float]:
    """Blend CASEJ/CASE2/ALMANAC point yields when PROCESS_BMA_ENABLED and method is bma|best."""
    if settings is None or not getattr(settings, "process_bma_enabled", False):
        return baseline_mean, projected_mean
    method = request.ensemble_process_method
    if method == "mean":
        return baseline_mean, projected_mean

    from models.process.bma import combine_predictions

    weights_path = getattr(settings, "process_bma_weights_path", Path("config/process_bma_weights.json"))

    def _casej_mean(climate: Tensor, static: Tensor) -> float | None:
        if not isinstance(model, CASEJSurrogate):
            return None
        return float(
            predict_scenario_yield_samples(model, climate, static, scenario_co2, 1).mean().item()
        )

    def _surrogate_mean(climate: Tensor, static: Tensor) -> float | None:
        if isinstance(model, CASEJSurrogate):
            return None
        region_id = _region_id_tensor(model, climate, lat=lat, lon=lon, country_code=country_code)
        teleconnection = feature_resolver.resolve_teleconnection(lat, lon, horizon)
        return float(
            predict_yield_samples(
                model,
                climate,
                static,
                1,
                region_id=region_id,
                teleconnection=teleconnection,
                lat=lat,
                lon=lon,
            ).mean().item()
        )

    ds_b = _climate_tensor_to_dataset(climate_baseline, year)
    ds_p = _climate_tensor_to_dataset(climate_projected, year)
    casej_b = _casej_mean(climate_baseline, static_cf) or _surrogate_mean(climate_baseline, static_cf)
    casej_p = _casej_mean(climate_projected, static_factual) or _surrogate_mean(
        climate_projected, static_factual
    )
    case2_b = _try_case2_yield_tonnes_ha(ds_b)
    case2_p = _try_case2_yield_tonnes_ha(ds_p)
    alm_b = _try_almanac_yield_tonnes_ha(ds_b)
    alm_p = _try_almanac_yield_tonnes_ha(ds_p)

    try:
        new_b = combine_predictions(
            casej=casej_b,
            case2=case2_b,
            almanac=alm_b,
            method=method,
            weights_path=weights_path,
        )
        new_p = combine_predictions(
            casej=casej_p,
            case2=case2_p,
            almanac=alm_p,
            method=method,
            weights_path=weights_path,
        )
    except ValueError:
        return baseline_mean, projected_mean
    return new_b, new_p


def _biotic_static_features(
    intervention_type: InterventionType | None,
    *,
    lat: float | None = None,
    lon: float | None = None,
    year: int | None = None,
    cssvd_prevalence_pct: float = 15.0,
    cssvd_tolerance: float = 1.0,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Farm static covariates for biotic loss (CRIG prevalence mock until raster wired)."""
    shade = ShadeSpecies.UNSHADED
    if intervention_type is not None:
        deltas = INTERVENTION_CLIMATE_DELTAS.get(intervention_type, {})
        raw_shade = deltas.get("shade_species")
        if raw_shade is not None:
            shade = ShadeSpecies(str(raw_shade))
    out: dict[str, Any] = {
        "cssvd_prevalence_pct": cssvd_prevalence_pct,
        "cssvd_tolerance": cssvd_tolerance,
        "shade_species": shade,
    }
    if lat is not None and lon is not None:
        out["lat"] = float(lat)
        out["lon"] = float(lon)
    if year is not None:
        out["year"] = int(year)
    if settings is not None:
        if getattr(settings, "enable_cssvd_landscape", False):
            ckpt = getattr(settings, "cssvd_landscape_checkpoint", None)
            if ckpt is not None and Path(ckpt).is_file():
                out["use_cssvd_landscape"] = True
                out["cssvd_landscape_checkpoint"] = str(ckpt)
    return out


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


def _region_id_tensor(
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    climate: Tensor,
    *,
    lat: float,
    lon: float,
    country_code: str | None,
) -> Tensor | None:
    if not isinstance(model, (YieldSurrogateV2, YieldSurrogateV2Teleconnection)):
        return None
    if country_code:
        rid = region_id_from_country_code(country_code)
    else:
        rid = region_id_from_latlon(lat, lon)
    return torch.tensor([rid], dtype=torch.long, device=climate.device)


def yield_model_forward(
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    climate: Tensor,
    static: Tensor,
    *,
    region_id: Tensor | None = None,
    teleconnection: dict[str, Any] | None = None,
    lat: float = 6.0,
    lon: float = -2.0,
) -> Tensor:
    """Dispatch v1/v2/teleconnection composite forward."""
    if isinstance(model, YieldSurrogateV2Teleconnection):
        return model(
            climate,
            static,
            region_id,
            teleconnection,
            lat=lat,
            lon=lon,
        )
    if isinstance(model, YieldSurrogateV2):
        return model(climate, static, region_id)
    return model(climate, static)


@torch.no_grad()
def predict_yield_samples(
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    climate: Tensor,
    static: Tensor,
    num_samples: int,
    *,
    region_id: Tensor | None = None,
    teleconnection: dict[str, Any] | None = None,
    lat: float = 6.0,
    lon: float = -2.0,
) -> Tensor:
    """Run stochastic forward passes; returns ``[num_samples]`` yields."""
    from api.telemetry import trace_span

    with trace_span("yield_surrogate.forward", num_samples=num_samples):
        return _predict_yield_samples_impl(
            model,
            climate,
            static,
            num_samples,
            region_id=region_id,
            teleconnection=teleconnection,
            lat=lat,
            lon=lon,
        )


def _predict_yield_samples_impl(
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    climate: Tensor,
    static: Tensor,
    num_samples: int,
    *,
    region_id: Tensor | None = None,
    teleconnection: dict[str, Any] | None = None,
    lat: float = 6.0,
    lon: float = -2.0,
) -> Tensor:
    was_training = model.training
    model.eval()
    samples = torch.stack(
        [
            yield_model_forward(
                model,
                climate,
                static,
                region_id=region_id,
                teleconnection=teleconnection,
                lat=lat,
                lon=lon,
            ).squeeze(0)
            for _ in range(num_samples)
        ],
        dim=0,
    )
    if was_training:
        model.train()
    return samples


@torch.no_grad()
def predict_scenario_yield_samples(
    model: CASEJSurrogate,
    climate: Tensor,
    static: Tensor,
    co2_ppm: float,
    num_samples: int,
) -> Tensor:
    """MC-dropout samples for CASEJ surrogate with explicit CO2 (t/ha)."""
    was_training = model.training
    model.eval()
    co2 = torch.tensor([co2_ppm], dtype=climate.dtype, device=climate.device)
    samples = torch.stack(
        [model(climate, static, co2_ppm=co2).squeeze(0) for _ in range(num_samples)],
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


def _mcd_avoided_loss_interval(
    samples_cf: Tensor,
    samples_factual: Tensor,
    farm_size_ha: float,
) -> tuple[float, float]:
    """90% interval from paired MC yield samples (5th–95th percentile on avoided tonnes)."""
    delta_per_ha = (samples_factual - samples_cf).cpu().numpy()
    avoided_per_ha = np.maximum(delta_per_ha, 0.0)
    avoided_loss_samples = avoided_per_ha * farm_size_ha
    return (
        float(np.percentile(avoided_loss_samples, 5.0)),
        float(np.percentile(avoided_loss_samples, 95.0)),
    )


@torch.no_grad()
def _simulate_cqr(
    cqr_model: QuantileYieldSurrogate,
    calibrator: ConformalCalibrator,
    climate_cf: Tensor,
    static_cf: Tensor,
    climate_factual: Tensor,
    static_factual: Tensor,
    *,
    biotic_cf: float,
    biotic_factual: float,
    farm_size_ha: float,
    current_yield: float,
    yield_blend_weight: float,
) -> tuple[float, float, float, float, float | None]:
    """
    CQR yield intervals for baseline vs projected and conservative avoided-loss bounds.

    Returns
    -------
    baseline_yield, projected_yield, ci_lower, ci_upper, empirical_coverage
    """
    base_iv = calibrator.predict_interval(cqr_model, (climate_cf, static_cf))
    fact_iv = calibrator.predict_interval(cqr_model, (climate_factual, static_factual))

    baseline_yield = _blend_yield(base_iv.median * biotic_cf, current_yield, yield_blend_weight)
    projected_yield = _blend_yield(fact_iv.median * biotic_factual, current_yield, yield_blend_weight)

    ci_lower = max(
        0.0,
        (fact_iv.lower * biotic_factual - base_iv.upper * biotic_cf) * farm_size_ha,
    )
    ci_upper = max(
        0.0,
        (fact_iv.upper * biotic_factual - base_iv.lower * biotic_cf) * farm_size_ha,
    )
    return (
        baseline_yield,
        projected_yield,
        ci_lower,
        ci_upper,
        calibrator.empirical_coverage,
    )


def _optional_eudr_status(
    request: SimulateInterventionRequest | SimulateScenarioRequest,
    settings: Any,
) -> Any:
    if not getattr(request, "farm_polygon", None):
        return None
    from api.eudr import evaluate_eudr_status

    return evaluate_eudr_status(request.farm_polygon, settings=settings)


def simulate_intervention(
    request: SimulateInterventionRequest,
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    feature_resolver: FarmFeatureResolver,
    *,
    num_samples: int = 50,
    yield_blend_weight: float = 0.0,
    climate_year: int | None = None,
    conformal: ConformalPredictor | None = None,
    uq_method: UQMethod = "mcd",
    cqr_model: QuantileYieldSurrogate | None = None,
    cqr_calibrator: ConformalCalibrator | None = None,
    settings: Any = None,
) -> SimulateInterventionResponse:
    """
    Predict counterfactual vs factual yield and compute avoided loss + financial impact.

    Uses :class:`~api.feature_resolver.FarmFeatureResolver` for ERA5/static features
    (``USE_REAL_FEATURES=true`` → Zarr/cache/GEE; ``false`` → ``api.geo_mock`` for tests)
    and paired Monte Carlo samples for a 90% confidence interval on avoided loss.
    """
    if yield_blend_weight > 0.0:
        log.warning(
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

    use_cqr = (
        uq_method == "cqr"
        and cqr_model is not None
        and cqr_calibrator is not None
    )

    samples_cf: Tensor | None = None
    samples_factual: Tensor | None = None
    region_id = _region_id_tensor(
        model,
        climate_cf,
        lat=lat,
        lon=lon,
        country_code=request.country_code,
    )
    teleconnection = feature_resolver.resolve_teleconnection(lat, lon, year)
    if not use_cqr:
        samples_cf = predict_yield_samples(
            model,
            climate_cf,
            static_cf,
            num_samples,
            region_id=region_id,
            teleconnection=teleconnection,
            lat=lat,
            lon=lon,
        )
        samples_factual = predict_yield_samples(
            model,
            climate_factual,
            static_factual,
            num_samples,
            region_id=region_id,
            teleconnection=teleconnection,
            lat=lat,
            lon=lon,
        )

    ds_cf = _climate_tensor_to_dataset(climate_cf, year)
    ds_factual = _climate_tensor_to_dataset(climate_factual, year)
    biotic_static_cf = _biotic_static_features(
        None, lat=lat, lon=lon, year=year, settings=settings
    )
    biotic_static_factual = _biotic_static_features(
        request.intervention_type, lat=lat, lon=lon, year=year, settings=settings
    )
    biotic_cf = apply_biotic_losses(1.0, ds_cf, biotic_static_cf)
    biotic_factual = apply_biotic_losses(1.0, ds_factual, biotic_static_factual)
    biotic_cf_frac = float(biotic_cf["surviving_fraction"])
    biotic_fact_frac = float(biotic_factual["surviving_fraction"])

    empirical_coverage: float | None = None
    if use_cqr:
        baseline_yield, projected_yield, ci_lower, ci_upper, empirical_coverage = _simulate_cqr(
            cqr_model,
            cqr_calibrator,
            climate_cf,
            static_cf,
            climate_factual,
            static_factual,
            biotic_cf=biotic_cf_frac,
            biotic_factual=biotic_fact_frac,
            farm_size_ha=request.farm_size_ha,
            current_yield=request.current_yield,
            yield_blend_weight=yield_blend_weight,
        )
    else:
        assert samples_cf is not None and samples_factual is not None
        samples_cf = samples_cf * biotic_cf_frac
        samples_factual = samples_factual * biotic_fact_frac

        mc_baseline = float(samples_cf.mean().item())
        mc_projected_raw = float(samples_factual.mean().item())

        baseline_yield = _blend_yield(mc_baseline, request.current_yield, yield_blend_weight)
        projected_yield = _blend_yield(mc_projected_raw, request.current_yield, yield_blend_weight)

        ci_lower, ci_upper = _mcd_avoided_loss_interval(
            samples_cf,
            samples_factual,
            request.farm_size_ha,
        )

    avoided_loss_tonnes = max(0.0, (projected_yield - baseline_yield) * request.farm_size_ha)

    fin = calculate_financial_impact(
        avoided_loss_tonnes,
        currency=request.currency,
        pricing_basis=request.pricing_basis,
        farm_gate=request.farm_gate,
        country_code=request.country_code,
        lat=lat,
        lon=lon,
        cocoa_price_usd=request.cocoa_price_usd,
        ci_low_tonnes=ci_lower,
        ci_high_tonnes=ci_upper,
    )
    financial_impact_usd = fin.usd.point

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

    eudr_status = _optional_eudr_status(request, settings) if settings is not None else None

    sensitivity_bounds = None
    if getattr(request, "include_sensitivity", False) and settings is not None:
        from api.causal_sensitivity import compute_sensitivity_bounds

        sensitivity_bounds = compute_sensitivity_bounds(settings)

    mediation_block = None
    decompose = getattr(request, "decompose_mediators", None)
    if decompose and samples_cf is not None and samples_factual is not None:
        from api.mediation import compute_intervention_mediation

        n_boot = 200
        if settings is not None:
            n_boot = int(getattr(settings, "mediation_n_bootstrap", 200))
        biotic_base = biotic_cf
        biotic_proj = biotic_factual
        mediation_block = compute_intervention_mediation(
            request,
            samples_cf=samples_cf,
            samples_factual=samples_factual,
            ds_cf=ds_cf,
            ds_factual=ds_factual,
            biotic_baseline=biotic_base,
            biotic_projected=biotic_proj,
            decompose_mediators=decompose,
            n_bootstrap=n_boot,
        )

    from api import metrics as prom_metrics

    prom_metrics.observe_avoided_loss("simulate_intervention", avoided_loss_tonnes)

    return SimulateInterventionResponse(
        baseline_yield_tonnes_per_ha=baseline_yield,
        projected_yield_tonnes_per_ha=projected_yield,
        avoided_loss_tonnes=avoided_loss_tonnes,
        financial_impact_usd=financial_impact_usd,
        financial_impact=financial_impact_to_schema(fin),
        confidence_interval=ConfidenceInterval(
            avoided_loss_tonnes=AvoidedLossInterval(
                lower=ci_lower,
                upper=ci_upper,
                level=0.9,
            ),
            method="cqr" if use_cqr else "mcd",
            empirical_coverage=empirical_coverage,
        ),
        conformal_interval=conformal_block,
        biotic_loss_attribution={
            "baseline": _biotic_response_block(biotic_cf),
            "projected": _biotic_response_block(biotic_factual),
        },
        eudr_status=eudr_status,
        sensitivity_bounds=sensitivity_bounds,
        mediation=mediation_block,
    )


@torch.no_grad()
def simulate_climate_attribution(
    request: SimulateClimateAttributionRequest,
    model: YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    feature_resolver: FarmFeatureResolver,
    *,
    counterfactual_zarr_path: Path,
    num_samples: int = 50,
    yield_blend_weight: float = 0.0,
    climate_year: int | None = None,
) -> SimulateClimateAttributionResponse:
    """
    Separate climate-change-driven yield loss from intervention avoided loss.

    Compares yields under factual ERA5 vs ATTRICI counterfactual (no-anthropogenic-forcing
    world), then adds intervention uplift from :func:`simulate_intervention`.
    """
    if not counterfactual_zarr_path.is_dir():
        raise ValueError(
            f"Counterfactual ERA5 Zarr not found at {counterfactual_zarr_path}. "
            "Pre-compute via scripts/run_attrici_civ_ghana.py."
        )

    lat = request.farm_location.lat
    lon = request.farm_location.lon
    year = int(climate_year or 2023)

    climate_factual_tensor = feature_resolver.resolve_climate(lat, lon, year)
    static_base = feature_resolver.resolve_static_with_galileo(lat, lon, year)
    static_cf = _encode_static(
        static_base,
        current_yield=request.current_yield,
        intervention_type=None,
    )

    cf_ds = xr.open_zarr(counterfactual_zarr_path, consolidated=False)
    era5_path = getattr(feature_resolver.config, "era5_zarr_path", None)
    if era5_path is not None and Path(era5_path).is_dir():
        factual_ds = xr.open_zarr(era5_path, consolidated=False)
    else:
        factual_ds = _climate_tensor_to_dataset(climate_factual_tensor, year)
    cf_daily = extract_daily_climate_11ch(
        cf_ds,
        lat,
        lon,
        year,
        factual_reference=factual_ds,
    )
    climate_cf_world = torch.from_numpy(cf_daily).unsqueeze(0)

    region_id = _region_id_tensor(
        model,
        climate_factual_tensor,
        lat=lat,
        lon=lon,
        country_code=request.country_code,
    )
    teleconnection = feature_resolver.resolve_teleconnection(lat, lon, year)
    samples_f = predict_yield_samples(
        model,
        climate_factual_tensor,
        static_cf,
        num_samples,
        region_id=region_id,
        teleconnection=teleconnection,
        lat=lat,
        lon=lon,
    )
    samples_cf = predict_yield_samples(
        model,
        climate_cf_world,
        static_cf,
        num_samples,
        region_id=region_id,
        teleconnection=teleconnection,
        lat=lat,
        lon=lon,
    )

    y_f = float(_blend_mc_numpy(samples_f.detach().cpu().numpy(), request.current_yield, yield_blend_weight).mean())
    y_cf = float(_blend_mc_numpy(samples_cf.detach().cpu().numpy(), request.current_yield, yield_blend_weight).mean())
    attributed_per_ha = max(0.0, y_cf - y_f)

    intervention = simulate_intervention(
        request,
        model,
        feature_resolver,
        num_samples=num_samples,
        yield_blend_weight=yield_blend_weight,
        climate_year=year,
    )
    intervention_avoided = float(intervention.avoided_loss_tonnes)
    total_avoided = attributed_per_ha * request.farm_size_ha + intervention_avoided

    fin = calculate_financial_impact(
        total_avoided,
        currency=request.currency,
        pricing_basis=request.pricing_basis,
        farm_gate=request.farm_gate,
        country_code=request.country_code,
        lat=lat,
        lon=lon,
        cocoa_price_usd=request.cocoa_price_usd,
    )

    return SimulateClimateAttributionResponse(
        factual_yield_tonnes_per_ha=y_f,
        counterfactual_yield_tonnes_per_ha=y_cf,
        attributed_loss_tonnes_per_ha=attributed_per_ha,
        intervention_avoided_loss_tonnes=intervention_avoided,
        total_avoided_loss_tonnes=total_avoided,
        climate_reference_year=year,
        financial_impact_usd=fin.usd.point,
        financial_impact=financial_impact_to_schema(fin),
    )


@torch.no_grad()
def simulate_scenario(
    request: SimulateScenarioRequest,
    model: CASEJSurrogate | YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection,
    feature_resolver: FarmFeatureResolver,
    *,
    historical_zarr_path: Path,
    cmip6_zarr_path: Path,
    num_samples: int = 50,
    yield_blend_weight: float = 0.0,
    climate_year: int | None = None,
    settings: Any = None,
    cqr_model: QuantileYieldSurrogate | None = None,
    cqr_calibrator: ConformalCalibrator | None = None,
    scenario_conformal_store: Any = None,
    drift_store: Any = None,
) -> SimulateScenarioResponse:
    """
    Future-climate avoided loss using CMIP6 delta-change on ERA5 (`ScenarioBuilder`) +
    paired Monte Carlo forwards through :class:`~models.casej_surrogate.CASEJSurrogate`
    with SSP-specific CO2 (ppm).

    Baseline is SSP-conditioned climate **without** intervention encoding; projected applies
    the usual mechanistic intervention deltas on the adjusted climate tensor.
    """
    if yield_blend_weight > 0.0:
        log.warning(
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
    downscaling = request.downscaling_method
    corrdiff_n: int | None = None

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

    scenario_co2 = co2_ppm_for_ssp(request.scenario, horizon)

    if downscaling == "linear_delta":
        builder = ScenarioBuilder(
            str(historical_zarr_path.resolve()),
            str(cmip6_zarr_path.resolve()),
        )
        ds_scenario = builder.build_scenario(request.scenario, window)
        climate_ensemble = [climate_tensor_from_dataset_point(ds_scenario, lat, lon, year)]
    elif downscaling == "neuralgcm":
        if settings is None or not getattr(settings, "neuralgcm_enabled", False):
            raise ValueError("neuralgcm downscaling requires NEURALGCM_ENABLED=true")
        from counterfactual.neuralgcm_runner import emulate_era5_point

        ds_ng = emulate_era5_point(lat=lat, lon=lon, start=window[0], end=window[1])
        climate_ensemble = [climate_tensor_from_dataset_point(ds_ng, lat, lon, horizon)]
    elif downscaling == "ace2_era5":
        if settings is None or not getattr(settings, "ace2_era5_enabled", False):
            raise ValueError("ace2_era5 downscaling requires ACE2_ERA5_ENABLED=true")
        from counterfactual.ace2_era5_runner import emulate_era5_ace2

        ds_ace = emulate_era5_ace2(lat=lat, lon=lon, start=window[0], end=window[1])
        climate_ensemble = [climate_tensor_from_dataset_point(ds_ace, lat, lon, horizon)]
    elif downscaling == "corrdiff":
        processed_dir = (
            settings.corrdiff_processed_dir
            if settings is not None
            else historical_zarr_path.parent
        )
        region = resolve_region(lat, lon)
        cache = corrdiff_cache_path(processed_dir, request.scenario, horizon, region)
        if not cache.is_dir():
            if settings is not None and settings.corrdiff_allow_inline:
                downscaler = CorrDiffCMIP6Downscaler(
                    experiment_id=request.scenario,  # type: ignore[arg-type]
                    source_id=settings.corrdiff_source_id,
                    variant_label=settings.corrdiff_variant_label,
                    number_of_samples=settings.corrdiff_number_of_samples,
                    solver=settings.corrdiff_solver,
                    sampler_type=settings.corrdiff_sampler_type,
                    region=region,
                    historical_zarr_path=historical_zarr_path,
                    cmip6_zarr_path=cmip6_zarr_path,
                )
                downscaler.downscale_horizon_year(
                    horizon, list(DEFAULT_OUTPUT_VARIABLES)
                )
                downscaler.to_zarr(cache)
            else:
                raise ValueError(corrdiff_cache_missing_message(cache, request.scenario, horizon, region))
        try:
            climate_ensemble = load_corrdiff_scenario_ensemble(
                cache_path=cache, lat=lat, lon=lon, year=horizon
            )
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        corrdiff_n = len(climate_ensemble)
    else:
        raise ValueError(f"Unknown downscaling_method: {downscaling}")

    climate_baseline = climate_ensemble[0].clone()
    climate_projected = _apply_intervention_climate(climate_ensemble[0], request.intervention_type)

    use_casej = (
        settings is not None
        and getattr(settings, "scenario_yield_backend", "v2_teleconnection") == "casej"
    )
    baseline_draws: list[np.ndarray] = []
    projected_draws: list[np.ndarray] = []
    for climate_scenario in climate_ensemble:
        climate_b = climate_scenario.clone()
        climate_p = _apply_intervention_climate(climate_scenario, request.intervention_type)
        for tensor in (climate_b, climate_p):
            tensor[..., CLIMATE_IDX["co2_ppm"]] = scenario_co2

        if use_casej and isinstance(model, CASEJSurrogate):
            samples_cf = predict_scenario_yield_samples(
                model, climate_b, static_cf, scenario_co2, num_samples
            )
            samples_factual = predict_scenario_yield_samples(
                model, climate_p, static_factual, scenario_co2, num_samples
            )
        else:
            region_id = _region_id_tensor(
                model,
                climate_b,
                lat=lat,
                lon=lon,
                country_code=request.country_code,
            )
            teleconnection = feature_resolver.resolve_teleconnection(lat, lon, horizon)
            samples_cf = predict_yield_samples(
                model,
                climate_b,
                static_cf,
                num_samples,
                region_id=region_id,
                teleconnection=teleconnection,
                lat=lat,
                lon=lon,
            )
            samples_factual = predict_yield_samples(
                model,
                climate_p,
                static_factual,
                num_samples,
                region_id=region_id,
                teleconnection=teleconnection,
                lat=lat,
                lon=lon,
            )
        baseline_draws.append(samples_cf.detach().cpu().numpy().reshape(-1))
        projected_draws.append(samples_factual.detach().cpu().numpy().reshape(-1))

    baseline_np = np.concatenate(baseline_draws)
    projected_np = np.concatenate(projected_draws)
    for tensor in (climate_baseline, climate_projected):
        tensor[..., CLIMATE_IDX["co2_ppm"]] = scenario_co2

    baseline_blended = _blend_mc_numpy(baseline_np, request.current_yield, yield_blend_weight)
    projected_blended = _blend_mc_numpy(projected_np, request.current_yield, yield_blend_weight)

    b_mean, b_p10, b_p90 = _mean_p10_p90(baseline_blended)
    p_mean, p_p10, p_p90 = _mean_p10_p90(projected_blended)

    b_mean_bma, p_mean_bma = _apply_process_bma_to_means(
        request=request,
        settings=settings,
        model=model,
        baseline_mean=b_mean,
        projected_mean=p_mean,
        climate_baseline=climate_baseline,
        climate_projected=climate_projected,
        static_cf=static_cf,
        static_factual=static_factual,
        scenario_co2=scenario_co2,
        year=year,
        lat=lat,
        lon=lon,
        country_code=request.country_code,
        feature_resolver=feature_resolver,
        horizon=horizon,
    )
    if abs(b_mean) > 1e-9 and b_mean_bma != b_mean:
        baseline_blended = baseline_blended * (b_mean_bma / b_mean)
        b_mean = b_mean_bma
    if abs(p_mean) > 1e-9 and p_mean_bma != p_mean:
        projected_blended = projected_blended * (p_mean_bma / p_mean)
        p_mean = p_mean_bma
    b_p10, b_p90 = float(np.percentile(baseline_blended, 10)), float(np.percentile(baseline_blended, 90))
    p_p10, p_p90 = float(np.percentile(projected_blended, 10)), float(np.percentile(projected_blended, 90))

    avoided_arr = np.maximum(projected_blended - baseline_blended, 0.0) * request.farm_size_ha
    a_mean, a_p10, a_p90 = _mean_p10_p90(avoided_arr)
    a_mean, a_p10, a_p90 = (max(0.0, a_mean), max(0.0, a_p10), max(0.0, a_p90))

    ds_cf = _climate_tensor_to_dataset(climate_baseline, year)
    ds_factual = _climate_tensor_to_dataset(climate_projected, year)
    biotic_static_cf = _biotic_static_features(
        None, lat=lat, lon=lon, year=year, settings=settings
    )
    biotic_static_factual = _biotic_static_features(
        request.intervention_type, lat=lat, lon=lon, year=year, settings=settings
    )
    biotic_cf = apply_biotic_losses(1.0, ds_cf, biotic_static_cf)
    biotic_factual = apply_biotic_losses(1.0, ds_factual, biotic_static_factual)
    biotic_cf_frac = float(biotic_cf["surviving_fraction"])
    biotic_fact_frac = float(biotic_factual["surviving_fraction"])

    confidence_interval: ConfidenceInterval | None = None
    drift_alarm = None
    drift_status = None
    fin_ci_low, fin_ci_high = a_p10, a_p90
    if cqr_model is not None and settings is not None:
        conformal_result = apply_scenario_conformal(
            request,
            cqr_model=cqr_model,
            cqr_calibrator=cqr_calibrator,
            store=scenario_conformal_store,
            drift_store=drift_store,
            settings=settings,
            climate_baseline=climate_baseline,
            climate_projected=climate_projected,
            static_cf=static_cf,
            static_factual=static_factual,
            biotic_cf_frac=biotic_cf_frac,
            biotic_fact_frac=biotic_fact_frac,
        )
        if conformal_result is not None:
            fin_ci_low = conformal_result.ci_lower
            fin_ci_high = conformal_result.ci_upper
            confidence_interval = conformal_result.confidence_interval
            drift_alarm = conformal_result.drift_alarm
            drift_status = conformal_result.drift_status

    fin = calculate_financial_impact(
        a_mean,
        currency=request.currency,
        pricing_basis=request.pricing_basis,
        farm_gate=request.farm_gate,
        country_code=request.country_code,
        lat=lat,
        lon=lon,
        cocoa_price_usd=request.cocoa_price_usd,
        ci_low_tonnes=fin_ci_low,
        ci_high_tonnes=fin_ci_high,
    )

    eudr_status = _optional_eudr_status(request, settings) if settings is not None else None

    from api import metrics as prom_metrics

    prom_metrics.observe_avoided_loss("simulate_scenario", float(a_mean))

    return SimulateScenarioResponse(
        scenario=request.scenario,
        horizon_year=request.horizon_year,
        downscaling_method=downscaling,
        corrdiff_samples_used=corrdiff_n,
        climate_reference_year=year if downscaling == "linear_delta" else horizon,
        baseline_yield_tonnes_per_ha=YieldUncertaintyBand(mean=b_mean, p10=b_p10, p90=b_p90),
        projected_yield_tonnes_per_ha=YieldUncertaintyBand(mean=p_mean, p10=p_p10, p90=p_p90),
        avoided_loss_tonnes=AvoidedLossUncertaintyBand(mean=a_mean, p10=a_p10, p90=a_p90),
        financial_impact_usd_mean=fin.usd.point,
        financial_impact=financial_impact_to_schema(fin),
        confidence_interval=confidence_interval,
        drift_alarm=drift_alarm,
        drift_status=drift_status,
        eudr_status=eudr_status,
    )
