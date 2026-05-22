"""Optional OpenTelemetry tracing (disabled unless OTEL_ENABLED=true)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

_tracer: Any = None
_enabled: bool = False


def is_enabled() -> bool:
    return _enabled


def configure_tracing(
    *,
    otlp_endpoint: str,
    service_name: str,
    service_version: str,
    environment: str,
) -> Any:
    """Configure OTLP exporter and global tracer; returns tracer or None."""
    global _tracer, _enabled
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)
    parsed = urlparse(otlp_endpoint)
    endpoint = otlp_endpoint
    if parsed.scheme in ("http", "https") and not parsed.path.startswith(":"):
        host = parsed.hostname or "localhost"
        port = parsed.port or (4317 if parsed.scheme == "http" else 443)
        endpoint = f"{host}:{port}"
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("cocoa-model-api")
    _enabled = True
    return _tracer


def get_tracer() -> Any:
    return _tracer


def current_trace_id_hex() -> str | None:
    """32-char hex trace id from active span, or None."""
    if not _enabled:
        return None
    from opentelemetry import trace

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[None]:
    """Child span when tracing enabled; no-op otherwise."""
    if not _enabled or _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))
        yield


def instrument_fastapi(app: Any) -> None:
    if not _enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


__all__ = [
    "configure_tracing",
    "current_trace_id_hex",
    "get_tracer",
    "instrument_fastapi",
    "is_enabled",
    "trace_span",
]
