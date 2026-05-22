"""Prometheus /metrics exposure and auth."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.metrics as prom


def _settings(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "prometheus_enabled": False,
        "metrics_auth_token": None,
        "prometheus_metrics_path": "/metrics",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_metrics_disabled_by_default() -> None:
    app = FastAPI()
    prom.setup_metrics(app, _settings(prometheus_enabled=False))
    client = TestClient(app)
    assert client.get("/metrics").status_code == 404


def test_metrics_enabled_open_and_auth() -> None:
    """Single instrumented app per process (prometheus registry is global)."""
    app = FastAPI()
    prom.setup_metrics(
        app,
        _settings(prometheus_enabled=True, metrics_auth_token="secret"),
    )
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert "cocoa_inference_latency_seconds" in ok.text
