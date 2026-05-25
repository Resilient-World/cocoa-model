from __future__ import annotations

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api.main import app, verify_api_key
from tests.conftest import API_KEY_HEADERS


def test_health_is_public() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_protected_post_requires_api_key() -> None:
    with TestClient(app) as client:
        response = client.post("/simulate-intervention", json={})
    assert response.status_code == 422


def test_protected_post_rejects_wrong_api_key() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/simulate-intervention",
            json={},
            headers={"x-api-key": "wrong-key"},
        )
    assert response.status_code == 401


def test_protected_post_allows_valid_api_key_to_reach_validation() -> None:
    with TestClient(app) as client:
        response = client.post("/simulate-intervention", json={}, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_required_post_routes_have_api_key_dependency() -> None:
    protected_paths = {
        "/simulate-intervention",
        "/simulate-scenario",
        "/compliance/dds",
        "/learn-policy-rules",
    }
    route_dependencies = {
        route.path: {dep.call for dep in route.dependant.dependencies}
        for route in app.routes
        if isinstance(route, APIRoute) and route.path in protected_paths
    }
    assert set(route_dependencies) == protected_paths
    for dependencies in route_dependencies.values():
        assert verify_api_key in dependencies


def test_cors_allows_configured_origin() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/simulate-intervention",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key, content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8000"
    assert "x-api-key" in response.headers["access-control-allow-headers"].lower()
