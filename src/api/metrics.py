"""Optional Prometheus metrics (disabled unless PROMETHEUS_ENABLED=true)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

_enabled: bool = False
_metrics_path: str = "/metrics"

# Custom metrics (initialized in setup_metrics)
INFERENCE_LATENCY: Any = None
AVOIDED_LOSS: Any = None
CONFORMAL_COVERAGE: Any = None
DRIFT_SCORE: Any = None
EUDR_COMPLIANCE: Any = None
MEDIATION_RATIO: Any = None
SIMULATION_ERRORS: Any = None
POLICY_RANKING_TOTAL: Any = None


def is_enabled() -> bool:
    return _enabled


def _init_custom_metrics() -> None:
    global INFERENCE_LATENCY, AVOIDED_LOSS, CONFORMAL_COVERAGE, DRIFT_SCORE
    global EUDR_COMPLIANCE, MEDIATION_RATIO, SIMULATION_ERRORS, POLICY_RANKING_TOTAL
    if INFERENCE_LATENCY is not None:
        return
    from prometheus_client import Counter, Gauge, Histogram

    INFERENCE_LATENCY = Histogram(
        "cocoa_inference_latency_seconds",
        "End-to-end handler latency",
        ["endpoint", "model_version"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    AVOIDED_LOSS = Histogram(
        "cocoa_avoided_loss_tonnes_per_request",
        "Avoided loss tonnes per simulation request",
        ["endpoint"],
        buckets=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0),
    )
    CONFORMAL_COVERAGE = Gauge(
        "cocoa_conformal_coverage_running_avg",
        "Rolling conformal coverage by stratum",
        ["scenario", "horizon", "region"],
    )
    DRIFT_SCORE = Gauge(
        "cocoa_drift_score",
        "WCTM wealth (exp log-martingale) by model and region",
        ["model", "region"],
    )
    EUDR_COMPLIANCE = Counter(
        "cocoa_eudr_compliance_total",
        "EUDR screening outcomes",
        ["status"],
    )
    MEDIATION_RATIO = Gauge(
        "cocoa_mediation_nde_nie_ratio",
        "NIE/NDE ratio from mediation decomposition",
        ["intervention"],
    )
    SIMULATION_ERRORS = Counter(
        "cocoa_simulation_errors_total",
        "Simulation endpoint errors",
        ["error_class", "endpoint"],
    )
    POLICY_RANKING_TOTAL = Counter(
        "cocoa_policy_ranking_total",
        "Policy targeting endpoint calls",
        ["endpoint"],
    )


class MetricsAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token on /metrics when METRICS_AUTH_TOKEN is set."""

    def __init__(self, app: Any, *, token: str | None, metrics_path: str) -> None:
        super().__init__(app)
        self._token = token
        self._metrics_path = metrics_path.rstrip("/") or "/metrics"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path.rstrip("/") == self._metrics_path and self._token:
            auth = request.headers.get("authorization", "")
            expected = f"Bearer {self._token}"
            if auth != expected:
                return Response(status_code=401, content="Unauthorized")
        return await call_next(request)


def setup_metrics(app: FastAPI, settings: Any) -> None:
    """Enable instrumentator + custom metrics when settings.prometheus_enabled."""
    global _enabled, _metrics_path
    if not getattr(settings, "prometheus_enabled", False):
        return
    _init_custom_metrics()
    _metrics_path = getattr(settings, "prometheus_metrics_path", "/metrics") or "/metrics"
    token = getattr(settings, "metrics_auth_token", None)
    app.add_middleware(MetricsAuthMiddleware, token=token, metrics_path=_metrics_path)

    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", _metrics_path],
    ).instrument(app).expose(
        app,
        endpoint=_metrics_path,
        include_in_schema=False,
    )
    _enabled = True


def observe_inference_latency(
    endpoint: str,
    model_version: str,
    seconds: float,
) -> None:
    if _enabled and INFERENCE_LATENCY is not None:
        INFERENCE_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(seconds)


def observe_avoided_loss(endpoint: str, tonnes: float) -> None:
    if _enabled and AVOIDED_LOSS is not None:
        AVOIDED_LOSS.labels(endpoint=endpoint).observe(max(0.0, tonnes))


def set_conformal_coverage(scenario: str, horizon: str, region: str, value: float) -> None:
    if _enabled and CONFORMAL_COVERAGE is not None:
        CONFORMAL_COVERAGE.labels(scenario=scenario, horizon=horizon, region=region).set(value)


def set_drift_score(model: str, region: str, score: float) -> None:
    if _enabled and DRIFT_SCORE is not None:
        DRIFT_SCORE.labels(model=model, region=region).set(score)


def inc_eudr_status(status: str) -> None:
    if _enabled and EUDR_COMPLIANCE is not None:
        EUDR_COMPLIANCE.labels(status=status).inc()


def set_mediation_ratio(intervention: str, ratio: float) -> None:
    if _enabled and MEDIATION_RATIO is not None:
        MEDIATION_RATIO.labels(intervention=intervention).set(ratio)


def inc_simulation_error(error_class: str, endpoint: str) -> None:
    if _enabled and SIMULATION_ERRORS is not None:
        SIMULATION_ERRORS.labels(error_class=error_class, endpoint=endpoint).inc()


def inc_policy_endpoint(endpoint: str) -> None:
    if _enabled and POLICY_RANKING_TOTAL is not None:
        POLICY_RANKING_TOTAL.labels(endpoint=endpoint).inc()


__all__ = [
    "setup_metrics",
    "is_enabled",
    "observe_inference_latency",
    "observe_avoided_loss",
    "set_conformal_coverage",
    "set_drift_score",
    "inc_eudr_status",
    "set_mediation_ratio",
    "inc_simulation_error",
    "inc_policy_endpoint",
]
