"""Tests for POST /simulate-intervention."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from models.yield_surrogate import N_CLIMATE_CHANNELS, YieldSurrogateModel

VALID_PAYLOAD = {
    "farm_location": {"lat": 6.5, "lon": -1.2},
    "farm_size_ha": 5.0,
    "current_yield": 2.0,
    "intervention_type": "shade_trees",
    "cocoa_price_usd": 3200.0,
}

SITE_STATIC_DIM = 10
SEQUENCE_LENGTH = 365


class StubFeatureResolver:
    """Deterministic stand-in for :class:`~api.feature_resolver.FarmFeatureResolver`."""

    def resolve_climate(self, lat: float, lon: float, year: int) -> torch.Tensor:
        del lat, lon, year
        rng = np.random.default_rng(42)
        seasonal = np.sin(2 * np.pi * np.arange(SEQUENCE_LENGTH) / 365.0)
        tmax = 30.0 + 2.0 * seasonal + rng.normal(0, 0.2, SEQUENCE_LENGTH)
        tmin = tmax - 7.0
        climate = np.zeros((SEQUENCE_LENGTH, N_CLIMATE_CHANNELS), dtype=np.float32)
        climate[:, 0] = tmax
        climate[:, 1] = tmin
        climate[:, 2] = 0.5 * (tmax + tmin)
        climate[:, 3] = np.clip(rng.gamma(2, 3, SEQUENCE_LENGTH), 0, 50)
        climate[:, 4] = 15.0 + 2.0 * seasonal
        climate[:, 5] = 1.2
        climate[:, 6] = 3.5
        climate[:, 7] = 0.28
        climate[:, 8] = 2.0
        climate[:, 9] = 75.0
        climate[:, 10] = 415.0
        return torch.from_numpy(climate).unsqueeze(0)

    def resolve_static(self, lat: float, lon: float) -> torch.Tensor:
        del lat, lon
        static = np.zeros(SITE_STATIC_DIM, dtype=np.float32)
        static[0] = 150.0
        static[1] = 0.4
        return torch.from_numpy(static).unsqueeze(0)

    def resolve_static_with_galileo(self, lat: float, lon: float, year: int) -> torch.Tensor:
        return self.resolve_static(lat, lon)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        app.state.feature_resolver = StubFeatureResolver()
        yield test_client


def test_feature_resolver_climate_and_static_shapes() -> None:
    resolver = StubFeatureResolver()
    climate = resolver.resolve_climate(6.5, -1.2, 2023)
    static = resolver.resolve_static(6.5, -1.2)
    assert climate.shape == (1, SEQUENCE_LENGTH, N_CLIMATE_CHANNELS)
    assert static.shape == (1, SITE_STATIC_DIM)
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
        galileo_dim=0,
    )
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "avoided_loss_tonnes" in response.json()


def test_yield_surrogate_galileo_dim_backward_compat() -> None:
    model = YieldSurrogateModel(static_features=10, galileo_dim=0)
    assert model.static_features == 10
    model_g = YieldSurrogateModel(static_features=10, galileo_dim=32)
    assert model_g.static_features == 42
    climate = torch.randn(2, 365, 11)
    static = torch.randn(2, 42)
    out = model_g(climate, static)
    assert out.shape == (2,)
