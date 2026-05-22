"""OpenTelemetry helpers (optional extra)."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("opentelemetry.sdk", reason="observability extra not installed")

from api import telemetry


def test_trace_span_noop_when_disabled() -> None:
    telemetry._enabled = False  # noqa: SLF001
    telemetry._tracer = None  # noqa: SLF001
    with telemetry.trace_span("test.span", foo="bar"):
        pass


def test_configure_tracing_returns_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.reload(telemetry)
    tracer = telemetry.configure_tracing(
        otlp_endpoint="http://localhost:4317",
        service_name="test-api",
        service_version="0.0.0",
        environment="test",
    )
    assert tracer is not None
    assert telemetry.is_enabled()
    telemetry._enabled = False  # noqa: SLF001
    telemetry._tracer = None  # noqa: SLF001
