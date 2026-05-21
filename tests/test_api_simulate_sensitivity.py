"""Tests for optional DVDS sensitivity on POST /simulate-intervention."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.config import APISettings
from api.main import app
from api.schemas import SensitivityBounds
from models.yield_surrogate import N_CLIMATE_CHANNELS, YieldSurrogateModel
from tests.test_api_simulate import StubFeatureResolver, VALID_PAYLOAD


class _SensitivityStubResolver(StubFeatureResolver):
    def resolve_teleconnection(self, lat: float, lon: float, year: int):
        del lat, lon, year
        return None


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.state.settings = APISettings(use_real_features=False, enable_teleconnection=False)
    app.state.feature_resolver = _SensitivityStubResolver()
    app.state.yield_model = YieldSurrogateModel(
        sequence_length=365,
        climate_features=N_CLIMATE_CHANNELS,
        static_features=13,
        galileo_dim=0,
    )
    app.state.casej_model = MagicMock()
    app.state.conformal = None
    app.state.cqr_model = None
    app.state.cqr_calibrator = None
    app.state.scenario_conformal_store = None
    app.state.drift_store = None
    yield TestClient(app, raise_server_exceptions=True)


def test_simulate_without_sensitivity_omits_bounds(client: TestClient) -> None:
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data.get("sensitivity_bounds") is None


@patch("api.causal_sensitivity.compute_sensitivity_bounds")
def test_simulate_with_sensitivity_returns_bounds(
    mock_bounds,
    client: TestClient,
) -> None:
    mock_bounds.return_value = [
        SensitivityBounds(
            lambda_=1.25,
            ate_lower=-0.1,
            ate_upper=0.4,
            ci_lower=-0.2,
            ci_upper=0.5,
            tipping_point_lambda=1.8,
        ),
        SensitivityBounds(
            lambda_=2.0,
            ate_lower=-0.2,
            ate_upper=0.6,
            ci_lower=-0.35,
            ci_upper=0.75,
            tipping_point_lambda=1.8,
        ),
    ]
    payload = {**VALID_PAYLOAD, "include_sensitivity": True}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 200
    data = response.json()
    bounds = data["sensitivity_bounds"]
    assert bounds is not None
    assert len(bounds) == 2
    assert bounds[0]["lambda_"] == 1.25
    assert bounds[0]["tipping_point_lambda"] == 1.8
    mock_bounds.assert_called_once()
