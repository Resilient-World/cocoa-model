"""Tests for POST /simulate-scenario (CMIP6-adjusted climate + MC bands)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr
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

SITE_STATIC_DIM = 13
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


SCENARIO_PAYLOAD = {
    **VALID_PAYLOAD,
    "scenario": "ssp245",
    "horizon_year": 2050,
}


def _scenario_grid_dataset() -> xr.Dataset:
    """Minimal ERA5-schema grid (1×1) for a full 2023 calendar used by climate_tensor_from_dataset_point."""
    times = xr.date_range("2023-01-01", periods=365, freq="D")
    t = len(times)
    shape = (t, 1, 1)
    rng = np.random.default_rng(7)
    tmax = (30.0 + 0.01 * np.arange(t)).reshape(shape).astype(np.float32)
    tmin = (tmax - 7.0).astype(np.float32)
    tmean = (0.5 * (tmax + tmin)).astype(np.float32)
    precip = np.abs(rng.normal(3.0, 1.0, shape)).astype(np.float32)
    srad = np.full(shape, 15.0, dtype=np.float32)
    vpd_mean = np.full(shape, 1.2, dtype=np.float32)
    et0 = np.full(shape, 3.5, dtype=np.float32)
    sm_root = np.full(shape, 0.28, dtype=np.float32)
    wind10m = np.full(shape, 2.0, dtype=np.float32)
    rh_mean = np.full(shape, 75.0, dtype=np.float32)
    co2_ppm = np.full(shape, 420.0, dtype=np.float32)
    return xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), tmax),
            "tmin": (("time", "lat", "lon"), tmin),
            "tmean": (("time", "lat", "lon"), tmean),
            "precip": (("time", "lat", "lon"), precip),
            "srad": (("time", "lat", "lon"), srad),
            "vpd_mean": (("time", "lat", "lon"), vpd_mean),
            "et0": (("time", "lat", "lon"), et0),
            "sm_root": (("time", "lat", "lon"), sm_root),
            "wind10m": (("time", "lat", "lon"), wind10m),
            "rh_mean": (("time", "lat", "lon"), rh_mean),
            "co2_ppm": (("time", "lat", "lon"), co2_ppm),
        },
        coords={
            "time": times,
            "lat": np.array([6.0], dtype=np.float32),
            "lon": np.array([-1.2], dtype=np.float32),
        },
    )


@pytest.fixture
def scenario_client(tmp_path):
    hist = tmp_path / "era5_stub"
    cmip = tmp_path / "cmip6_stub"
    hist.mkdir()
    cmip.mkdir()
    with TestClient(app) as client:
        app.state.settings.era5_zarr_path = hist
        app.state.settings.cmip6_zarr_path = cmip
        app.state.feature_resolver = StubFeatureResolver()
        yield client


@patch("api.simulation.ScenarioBuilder")
def test_simulate_scenario_happy_path(mock_sb_cls: MagicMock, scenario_client: TestClient) -> None:
    inst = MagicMock()
    mock_sb_cls.return_value = inst
    inst.build_scenario.return_value = _scenario_grid_dataset()

    app.state.yield_model = YieldSurrogateModel(
        sequence_length=365,
        climate_features=11,
        static_features=13,
        galileo_dim=0,
    )

    response = scenario_client.post("/simulate-scenario", json=SCENARIO_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario"] == "ssp245"
    assert data["horizon_year"] == 2050
    assert data["climate_reference_year"] == app.state.settings.climate_reference_year

    for key in ("baseline_yield_tonnes_per_ha", "projected_yield_tonnes_per_ha"):
        block = data[key]
        assert set(block.keys()) == {"mean", "p10", "p90"}
        assert block["p10"] <= block["mean"] <= block["p90"]

    avoided = data["avoided_loss_tonnes"]
    assert avoided["p10"] <= avoided["mean"] <= avoided["p90"]
    price = SCENARIO_PAYLOAD["cocoa_price_usd"] * 0.72  # GHA farm-gate pass-through
    assert data["financial_impact_usd_mean"] == pytest.approx(
        avoided["mean"] * price, rel=1e-3
    )
    assert data["financial_impact"]["ghs"]["currency"] == "GHS"


def test_simulate_scenario_missing_cmip6_zarr_returns_400(scenario_client: TestClient, tmp_path) -> None:
    """Paths must exist as directories before ScenarioBuilder runs."""
    missing = tmp_path / "nowhere"
    assert not missing.exists()
    app.state.settings.cmip6_zarr_path = missing

    response = scenario_client.post("/simulate-scenario", json=SCENARIO_PAYLOAD)
    assert response.status_code == 400
    assert "CMIP6 Zarr" in response.json()["detail"]


def test_simulate_scenario_invalid_horizon(scenario_client: TestClient) -> None:
    payload = {**SCENARIO_PAYLOAD, "horizon_year": 2040}
    response = scenario_client.post("/simulate-scenario", json=payload)
    assert response.status_code == 422
