"""Tests for POST /simulate-intervention."""

import pytest
from fastapi.testclient import TestClient

from api.geo_mock import CLIMATE_FEATURES, STATIC_FEATURES, fetch_climate_and_soil
from api.main import app
from models.yield_surrogate import YieldSurrogateModel

VALID_PAYLOAD = {
    "farm_location": {"lat": 6.5, "lon": -1.2},
    "farm_size_ha": 5.0,
    "current_yield": 2.0,
    "intervention_type": "shade_trees",
    "cocoa_price_usd": 3200.0,
}


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_geo_mock_climate_and_static_shapes() -> None:
    climate, static = fetch_climate_and_soil(6.5, -1.2)
    assert climate.shape == (1, 365, CLIMATE_FEATURES)
    assert static.shape == (1, STATIC_FEATURES)
    assert static[0, 0].item() == pytest.approx(150.0)


def test_simulate_intervention_happy_path(client: TestClient) -> None:
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    baseline = data["baseline_yield_tonnes_per_ha"]
    projected = data["projected_yield_tonnes_per_ha"]
    avoided = data["avoided_loss_tonnes"]
    financial = data["financial_impact_usd"]
    ci = data["confidence_interval"]["avoided_loss_tonnes"]

    assert avoided == pytest.approx(
        max(0.0, projected - baseline) * VALID_PAYLOAD["farm_size_ha"],
        rel=1e-5,
    )
    assert financial == pytest.approx(avoided * VALID_PAYLOAD["cocoa_price_usd"], rel=1e-5)
    assert ci["level"] == 0.9
    assert ci["lower"] <= avoided <= ci["upper"]


def test_shade_trees_intervention_response_schema(client: TestClient) -> None:
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "baseline_yield_tonnes_per_ha",
        "projected_yield_tonnes_per_ha",
        "avoided_loss_tonnes",
        "financial_impact_usd",
        "confidence_interval",
    }


def test_validation_invalid_latitude(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "farm_location": {"lat": 95.0, "lon": -1.2}}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_validation_negative_farm_size(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "farm_size_ha": -1.0}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_validation_missing_cocoa_price(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "cocoa_price_usd"}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_validation_unknown_intervention(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "intervention_type": "unknown_intervention"}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_simulate_with_overridden_model(client: TestClient) -> None:
    """Deterministic small model still returns structured response."""
    app.state.yield_model = YieldSurrogateModel(
        sequence_length=365,
        climate_features=11,
        static_features=10,
    )
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "avoided_loss_tonnes" in response.json()
