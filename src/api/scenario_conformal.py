"""Online conformal avoided-loss intervals for POST /simulate-scenario."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from api.drift_monitoring import apply_drift_monitoring
from api.online_conformal_store import ConformalMethod, OnlineConformalStore, stratum_key
from api.schemas import (
    AvoidedLossInterval,
    ConfidenceInterval,
    DriftAlarmPayload,
    DriftStatus,
    SimulateScenarioRequest,
)
from data.cocoa_exposure import REGIONS, region_for_point
from models.cqr import ConformalCalibrator, QuantileYieldSurrogate

if TYPE_CHECKING:
    from api.config import APISettings
    from monitoring.drift_store import DriftStore

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConformalResult:
    ci_lower: float
    ci_upper: float
    confidence_interval: ConfidenceInterval
    drift_alarm: DriftAlarmPayload | None = None
    drift_status: DriftStatus | None = None


def resolve_region(lat: float, lon: float) -> str:
    """FDP region key for a farm coordinate; nearest-region fallback if offshore."""
    found = region_for_point(lat, lon)
    if found is not None:
        return found
    best_key = "ghana"
    best_dist = float("inf")
    for key, preset in REGIONS.items():
        clat = 0.5 * (preset.south + preset.north)
        clon = 0.5 * (preset.west + preset.east)
        dist = (lat - clat) ** 2 + (lon - clon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


@torch.no_grad()
def _raw_cqr_quantiles(
    cqr_model: QuantileYieldSurrogate,
    climate: Tensor,
    static: Tensor,
) -> tuple[float, float, float]:
    cqr_model.eval()
    q_pred = cqr_model(climate, static).detach().cpu().numpy()
    return float(q_pred[0, 0]), float(q_pred[0, 1]), float(q_pred[0, 2])


def _avoided_loss_bounds(
    base_lo: float,
    base_hi: float,
    fact_lo: float,
    fact_hi: float,
    *,
    biotic_cf: float,
    biotic_factual: float,
    farm_size_ha: float,
) -> tuple[float, float]:
    ci_lower = max(
        0.0,
        (fact_lo * biotic_factual - base_hi * biotic_cf) * farm_size_ha,
    )
    ci_upper = max(
        0.0,
        (fact_hi * biotic_factual - base_lo * biotic_cf) * farm_size_ha,
    )
    return ci_lower, ci_upper


def apply_scenario_conformal(
    request: SimulateScenarioRequest,
    *,
    cqr_model: QuantileYieldSurrogate,
    cqr_calibrator: ConformalCalibrator | None,
    store: OnlineConformalStore | None,
    drift_store: DriftStore | None = None,
    settings: APISettings | Any,
    climate_baseline: Tensor,
    climate_projected: Tensor,
    static_cf: Tensor,
    static_factual: Tensor,
    biotic_cf_frac: float,
    biotic_fact_frac: float,
) -> ScenarioConformalResult | None:
    """
    Compute avoided-loss conformal bounds, online update, and optional drift monitoring.

    Returns ``None`` when CQR artifacts are missing and method is not splittable.
    """
    method: ConformalMethod = getattr(settings, "conformal_method", "eci_integral")

    if method == "split_cqr":
        if cqr_calibrator is None:
            logger.warning("split_cqr requested but calibrator missing; skipping conformal CI")
            return None
        base_iv = cqr_calibrator.predict_interval(
            cqr_model, (climate_baseline, static_cf)
        )
        fact_iv = cqr_calibrator.predict_interval(
            cqr_model, (climate_projected, static_factual)
        )
        ci_lower, ci_upper = _avoided_loss_bounds(
            base_iv.lower,
            base_iv.upper,
            fact_iv.lower,
            fact_iv.upper,
            biotic_cf=biotic_cf_frac,
            biotic_factual=biotic_fact_frac,
            farm_size_ha=request.farm_size_ha,
        )
        return ScenarioConformalResult(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_interval=ConfidenceInterval(
                avoided_loss_tonnes=AvoidedLossInterval(
                    lower=ci_lower,
                    upper=ci_upper,
                    level=0.9,
                ),
                method="cqr",
                empirical_coverage=cqr_calibrator.empirical_coverage,
                coverage_running_avg=None,
            ),
        )

    if store is None:
        return None

    key = stratum_key(
        request.scenario,
        request.horizon_year,
        resolve_region(request.farm_location.lat, request.farm_location.lon),
        downscaling_method=request.downscaling_method,
    )
    updater = store.get_updater(key, method=method)

    b_lo, b_med, b_hi = _raw_cqr_quantiles(cqr_model, climate_baseline, static_cf)
    f_lo, f_med, f_hi = _raw_cqr_quantiles(cqr_model, climate_projected, static_factual)

    q_adj = float(updater.current_threshold)
    fact_adj_lo, _, fact_adj_hi = f_lo - q_adj, f_med, f_hi + q_adj

    observed_y = float(request.current_yield)
    score = float(
        ConformalCalibrator.conformity_scores(
            np.array([observed_y]),
            np.array([fact_adj_lo]),
            np.array([fact_adj_hi]),
        )[0]
    )
    covered = fact_adj_lo <= observed_y <= fact_adj_hi
    updater.update(score, covered=covered)
    store.save_after_update(key, updater, covered=covered, method=method)

    q_adj = float(updater.current_threshold)

    drift_alarm, drift_status, apply_inflation = apply_drift_monitoring(
        request,
        y_obs=observed_y,
        y_pred=f_med,
        interval_lo=fact_adj_lo,
        interval_hi=fact_adj_hi,
        drift_store=drift_store,
        conformal_store=store,
        settings=settings,
        climate_projected=climate_projected,
        static_factual=static_factual,
    )
    if apply_inflation:
        factor = float(getattr(settings, "drift_inflation_factor", 1.5))
        q_adj *= factor

    base_adj_lo, base_adj_hi = b_lo - q_adj, b_hi + q_adj
    fact_adj_lo, fact_adj_hi = f_lo - q_adj, f_hi + q_adj

    ci_lower, ci_upper = _avoided_loss_bounds(
        base_adj_lo,
        base_adj_hi,
        fact_adj_lo,
        fact_adj_hi,
        biotic_cf=biotic_cf_frac,
        biotic_factual=biotic_fact_frac,
        farm_size_ha=request.farm_size_ha,
    )

    method_label: str = method if method != "split_cqr" else "cqr"
    return ScenarioConformalResult(
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_interval=ConfidenceInterval(
            avoided_loss_tonnes=AvoidedLossInterval(
                lower=ci_lower,
                upper=ci_upper,
                level=0.9,
            ),
            method=method_label,  # type: ignore[arg-type]
            empirical_coverage=None,
            coverage_running_avg=store.coverage_running_avg(key),
        ),
        drift_alarm=drift_alarm,
        drift_status=drift_status,
    )


__all__ = ["ScenarioConformalResult", "apply_scenario_conformal", "resolve_region"]
