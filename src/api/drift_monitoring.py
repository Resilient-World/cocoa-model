"""WCTM drift monitoring hook for /simulate-scenario."""

from __future__ import annotations

import structlog

from typing import TYPE_CHECKING, Any

from torch import Tensor

from api.online_conformal_store import OnlineConformalStore, stratum_key
from api.schemas import DriftAlarmPayload, DriftStatus
from monitoring.drift_store import DriftStore
from monitoring.wctm import (
    covariate_nonconformity,
    score_from_yield_observation,
    sigma_from_interval,
)

if TYPE_CHECKING:
    from api.config import APISettings
    from api.schemas import SimulateScenarioRequest

log = structlog.get_logger(__name__)

FEATURE_DIM = 64


def _flatten_features(
    climate: Tensor | None,
    static: Tensor | None,
    *,
    max_dim: int = FEATURE_DIM,
) -> list[float]:
    parts: list[float] = []
    if climate is not None:
        arr = climate.detach().cpu().numpy().reshape(-1)
        parts.extend(float(x) for x in arr[: max_dim // 2])
    if static is not None:
        arr = static.detach().cpu().numpy().reshape(-1)
        parts.extend(float(x) for x in arr[: max_dim // 2])
    if not parts:
        return [0.0] * min(8, max_dim)
    if len(parts) < max_dim:
        parts = parts + [0.0] * (max_dim - len(parts))
    return parts[:max_dim]


def apply_drift_monitoring(
    request: SimulateScenarioRequest,
    *,
    y_obs: float,
    y_pred: float,
    interval_lo: float,
    interval_hi: float,
    drift_store: DriftStore | None,
    conformal_store: OnlineConformalStore | None,
    settings: APISettings | Any,
    climate_projected: Tensor | None = None,
    static_factual: Tensor | None = None,
) -> tuple[DriftAlarmPayload | None, DriftStatus | None, bool]:
    """
    Update WCTM / X-WCTM / CUSUM and return alarm payload, dashboard status, inflation flag.

    Returns ``(drift_alarm, drift_status, apply_inflation)`` where inflation is True
    when diagnosis is ``concept_shift`` and an alarm is active.
    """
    if drift_store is None or not getattr(settings, "drift_enabled", True):
        return None, None, False

    key = stratum_key(
        request.scenario,
        request.horizon_year,
        _region_from_request(request, settings),
        downscaling_method=request.downscaling_method,
    )
    sigma_t = sigma_from_interval(interval_lo, interval_hi)
    score = score_from_yield_observation(y_obs, y_pred, sigma_t)

    wctm = drift_store.get_wctm(key)
    x_wctm = drift_store.get_x_wctm(key)
    cusum = drift_store.get_cusum(key)

    st = drift_store.get_stratum_state(key)
    feat_vec = _flatten_features(climate_projected, static_factual)
    x_score, new_ema = covariate_nonconformity(
        feat_vec,
        st.feature_ema if st.feature_ema else None,
    )

    wctm.update(score, weight=1.0)
    x_wctm.update(x_score, weight=1.0)
    x_alarm = x_wctm.detect() is not None or x_wctm.log_martingale > x_wctm.log_threshold
    wctm.set_x_alarm_active(x_alarm)
    cusum.update(score)

    alarm = wctm.detect()
    diagnosis = wctm.diagnose() if alarm else "none"
    alarm_at = alarm.triggered_at if alarm else None

    drift_store.save_after_update(
        key,
        wctm=wctm,
        x_wctm=x_wctm,
        cusum=cusum,
        diagnosis=diagnosis,
        alarm_at=alarm_at,
        feature_ema=new_ema,
    )

    coverage_avg = None
    if conformal_store is not None:
        coverage_avg = conformal_store.coverage_running_avg(key)

    status_dict = drift_store.get_drift_status(key, coverage_running_avg=coverage_avg)
    drift_status = DriftStatus.model_validate(status_dict)

    drift_alarm: DriftAlarmPayload | None = None
    if alarm is not None:
        drift_alarm = DriftAlarmPayload(
            type=alarm.diagnosis,  # type: ignore[arg-type]
            log_martingale=alarm.log_martingale,
            triggered_at=alarm.triggered_at,
        )

    apply_inflation = bool(
        alarm is not None
        and diagnosis == "concept_shift"
        and getattr(settings, "drift_inflation_factor", 1.0) > 1.0
    )
    return drift_alarm, drift_status, apply_inflation


def _region_from_request(request: SimulateScenarioRequest, settings: Any) -> str:
    from api.scenario_conformal import resolve_region

    return resolve_region(request.farm_location.lat, request.farm_location.lon)


def get_drift_status_for_stratum(
    stratum: str,
    *,
    drift_store: DriftStore | None,
    conformal_store: OnlineConformalStore | None = None,
) -> DriftStatus:
    if drift_store is None:
        raise ValueError("Drift store not configured")
    parts = stratum.split(":")
    if len(parts) == 4 and parts[-1] == "corrdiff":
        parts = parts[:-1]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid stratum key (expected scenario:horizon:region or ...:corrdiff): {stratum}"
        )
    coverage_avg = None
    if conformal_store is not None:
        coverage_avg = conformal_store.coverage_running_avg(stratum)
    data = drift_store.get_drift_status(stratum, coverage_running_avg=coverage_avg)
    return DriftStatus.model_validate(data)


__all__ = [
    "apply_drift_monitoring",
    "get_drift_status_for_stratum",
]
