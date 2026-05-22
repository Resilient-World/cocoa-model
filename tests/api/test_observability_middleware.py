"""structlog trace_id binding via observability middleware."""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.observability_middleware import register_observability_middleware
from common.logging import trace_id_var


def test_trace_id_var_cleared_after_request() -> None:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str | None]:
        return {"trace_id": trace_id_var.get()}

    register_observability_middleware(app)
    client = TestClient(app)
    before = trace_id_var.get()
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert trace_id_var.get() == before
