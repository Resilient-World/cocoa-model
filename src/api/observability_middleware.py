"""Bind OTel trace_id and request context to structlog."""

from __future__ import annotations

import time
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api import metrics as prom_metrics
from api import telemetry
from common.logging import trace_id_var


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        trace_id = telemetry.current_trace_id_hex()
        token = None
        if trace_id:
            token = trace_id_var.set(trace_id)
            structlog.contextvars.bind_contextvars(trace_id=trace_id)

        endpoint = request.url.path
        model_version = "unknown"
        settings = getattr(request.app.state, "settings", None)
        if settings is not None:
            model_version = getattr(settings, "yield_surrogate_version", "unknown")

        structlog.contextvars.bind_contextvars(
            endpoint=endpoint,
            model_version=model_version,
        )

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            prom_metrics.inc_simulation_error("unhandled", endpoint)
            raise
        finally:
            elapsed = time.perf_counter() - start
            if endpoint.startswith("/simulate"):
                prom_metrics.observe_inference_latency(
                    endpoint, str(model_version), elapsed
                )
            if status_code >= 500:
                prom_metrics.inc_simulation_error(f"http_{status_code}", endpoint)
            structlog.contextvars.clear_contextvars()
            if token is not None:
                trace_id_var.reset(token)


def register_observability_middleware(app: Any) -> None:
    app.add_middleware(ObservabilityMiddleware)
