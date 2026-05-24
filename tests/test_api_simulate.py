"""Tests for POST /simulate-intervention (mock vs real feature paths)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from fastapi.testclient import TestClient

from api.config import APISettings
from api.feature_resolver import FarmFeatureResolver, FeatureResolverConfig
from api.main import app
from data.gedi_canopy import CanopyPointSample
from models.yield_surrogate import N_CLIMATE_CHANNELS, YieldSurrogateModel

VALID_PAYLOAD = {
    "farm_location": {"lat": 6.5, "lon": -1.2},
    "farm_size_ha": 5.0,
    "current_yield": 2.0,
    "intervention_type": "shade_trees",
    "cocoa_price_usd": 3200.0,
}

SITE_STATIC_DIM = 15
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

    def resolve_static(self, lat: float, lon: float, year: int | None = None) -> torch.Tensor:
        del lat, lon, year
        static = np.zeros(SITE_STATIC_DIM, dtype=np.float32)
        static[0] = 150.0
        static[1] = 0.4
        return torch.from_numpy(static).unsqueeze(0)

    def resolve_static_with_galileo(self, lat: float, lon: float, year: int) -> torch.Tensor:
        return self.resolve_static(lat, lon)

    def resolve_canopy(self, lat: float, lon: float, year: int) -> CanopyPointSample:
        del lat, lon, year
        return CanopyPointSample(
            canopy_height_m=12.0,
            canopy_cover_pct=40.0,
            agb_mg_ha=150.0,
            height_uncertainty_m=1.2,
            gedi_n_shots=8,
            source_attributions=["test"],
        )


def _write_minimal_features_cache(path, *, lat: float = 6.5, lon: float = -1.2) -> None:
    """Tiny features_cache.zarr for real-feature API tests."""
    import pandas as pd

    time = pd.date_range("2023-01-01", periods=SEQUENCE_LENGTH, freq="D")
    climate = np.zeros((SEQUENCE_LENGTH, N_CLIMATE_CHANNELS), dtype=np.float32)
    climate[:, 0] = 28.0
    climate[:, 1] = 22.0
    climate[:, 2] = 25.0
    climate[:, 3] = 5.0
    climate[:, 4] = 15.0
    climate[:, 5] = 1.0
    climate[:, 6] = 3.0
    climate[:, 7] = 0.25
    climate[:, 8] = 2.0
    climate[:, 9] = 75.0
    climate[:, 10] = 420.0

    ds = xr.Dataset(
        {
            "clay_pct": (("latitude", "longitude"), np.array([[25.0]], dtype=np.float32)),
            "sand_pct": (("latitude", "longitude"), np.array([[40.0]], dtype=np.float32)),
            "soc_gkg": (("latitude", "longitude"), np.array([[20.0]], dtype=np.float32)),
            "cec_cmolkg": (("latitude", "longitude"), np.array([[15.0]], dtype=np.float32)),
            "ph": (("latitude", "longitude"), np.array([[5.5]], dtype=np.float32)),
            "elevation_m": (("latitude", "longitude"), np.array([[220.0]], dtype=np.float32)),
            "slope_deg": (("latitude", "longitude"), np.array([[2.0]], dtype=np.float32)),
            "chirps_annual_mm": (("latitude", "longitude"), np.array([[1400.0]], dtype=np.float32)),
            "protected_dist_km": (("latitude", "longitude"), np.array([[12.0]], dtype=np.float32)),
            "cocoa_prob": (("latitude", "longitude"), np.array([[0.82]], dtype=np.float32)),
            "canopy_height_m": (("latitude", "longitude"), np.array([[12.0]], dtype=np.float32)),
            "agb_mg_ha": (("latitude", "longitude"), np.array([[150.0]], dtype=np.float32)),
            "climate": (
                ("latitude", "longitude", "day", "channel"),
                climate.reshape(1, 1, SEQUENCE_LENGTH, N_CLIMATE_CHANNELS),
            ),
        },
        coords={
            "latitude": [lat],
            "longitude": [lon],
            "day": np.arange(SEQUENCE_LENGTH),
            "channel": np.arange(N_CLIMATE_CHANNELS),
        },
    )
    ds.to_zarr(path, mode="w")


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


def test_geo_mock_path_when_use_real_features_false(tmp_path) -> None:
    """USE_REAL_FEATURES=false → deterministic geo_mock tensors."""
    resolver = FarmFeatureResolver(
        FeatureResolverConfig(
            use_real_features=False,
            cache_dir=tmp_path / "cache",
        )
    )
    climate = resolver.resolve_climate(6.5, -1.2, 2023)
    static = resolver.resolve_static(6.5, -1.2, year=2023)
    assert climate.shape == (1, SEQUENCE_LENGTH, N_CLIMATE_CHANNELS)
    assert static.shape == (1, SITE_STATIC_DIM)
    assert static[0, 0].item() == pytest.approx(150.0)
    climate2 = resolver.resolve_climate(6.5, -1.2, 2023)
    assert torch.allclose(climate, climate2)


def test_real_features_from_cache_zarr(tmp_path) -> None:
    """USE_REAL_FEATURES=true reads precomputed features_cache.zarr."""
    cache_path = tmp_path / "features_cache.zarr"
    _write_minimal_features_cache(cache_path, lat=6.5, lon=-1.2)

    resolver = FarmFeatureResolver(
        FeatureResolverConfig(
            use_real_features=True,
            features_cache_zarr_path=cache_path,
            era5_zarr_path=tmp_path / "missing_era5.zarr",
            cache_dir=tmp_path / "cache",
        )
    )
    climate = resolver.resolve_climate(6.52, -1.18, 2023)
    static = resolver.resolve_static(6.48, -1.22, year=2023)
    assert climate.shape == (1, SEQUENCE_LENGTH, N_CLIMATE_CHANNELS)
    assert static.shape == (1, SITE_STATIC_DIM)
    assert static[0, 9].item() == pytest.approx(0.82, rel=0.01)
    assert static[0, 0].item() > 40.0


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
    assert financial == pytest.approx(avoided * VALID_PAYLOAD["cocoa_price_usd"] * 0.72, rel=1e-3)
    assert "financial_impact" in data
    assert data["financial_impact"]["usd"]["currency"] == "USD"
    assert data["financial_impact"]["ghs"]["currency"] == "GHS"
    assert data["financial_impact"]["xof"]["currency"] == "XOF"
    assert ci["level"] == 0.9
    assert ci["lower"] <= avoided <= ci["upper"]
    assert data["confidence_interval"]["method"] in ("mcd", "cqr")


def test_simulate_with_real_feature_resolver(tmp_path) -> None:
    """End-to-end API call using features_cache (not StubFeatureResolver)."""
    cache_path = tmp_path / "features_cache.zarr"
    _write_minimal_features_cache(cache_path)

    settings = APISettings(
        use_real_features=True,
        features_cache_zarr_path=cache_path,
        era5_zarr_path=tmp_path / "no_era5.zarr",
        feature_cache_dir=tmp_path / "api_cache",
    )
    app.state.settings = settings
    app.state.feature_resolver = FarmFeatureResolver(
        FeatureResolverConfig(
            use_real_features=True,
            features_cache_zarr_path=cache_path,
            era5_zarr_path=settings.era5_zarr_path,
            cache_dir=settings.feature_cache_dir,
        )
    )

    with TestClient(app) as test_client:
        # Lifespan resets resolver; override after startup for this test.
        app.state.feature_resolver = FarmFeatureResolver(
            FeatureResolverConfig(
                use_real_features=True,
                features_cache_zarr_path=cache_path,
                era5_zarr_path=settings.era5_zarr_path,
                cache_dir=settings.feature_cache_dir,
            )
        )
        response = test_client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "avoided_loss_tonnes" in response.json()


def test_shade_trees_intervention_response_schema(client: TestClient) -> None:
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert {
        "baseline_yield_tonnes_per_ha",
        "projected_yield_tonnes_per_ha",
        "avoided_loss_tonnes",
        "financial_impact_usd",
        "financial_impact",
        "confidence_interval",
        "conformal_interval",
        "biotic_loss_attribution",
    }.issubset(set(data.keys()))


def test_exposure_canopy_endpoint(client: TestClient) -> None:
    response = client.post(
        "/exposure-canopy",
        json={"farm_location": {"lat": 6.5, "lon": -1.2}, "year": 2023},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["canopy_height_m"] == pytest.approx(12.0)
    assert data["canopy_cover_pct"] == pytest.approx(40.0)
    assert data["agb_mg_ha"] == pytest.approx(150.0)
    assert data["gedi_n_shots"] == 8


def test_price_parametric_endpoint(client: TestClient) -> None:
    response = client.post(
        "/price-parametric",
        json={
            "farm_location": VALID_PAYLOAD["farm_location"],
            "farm_size_ha": 5.0,
            "strike_t_per_ha": 1.2,
            "coverage_horizon_years": 1,
            "scenario": "baseline",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["loaded_premium_usd"] >= data["fair_premium_usd"] >= 0.0
    assert 0.0 <= data["basis_risk_r2"] <= 1.0


def test_simulate_intervention_quality_block(client: TestClient) -> None:
    response = client.post(
        "/simulate-intervention",
        json={**VALID_PAYLOAD, "include_quality": True},
    )
    assert response.status_code == 200
    quality = response.json()["quality"]
    assert 0.0 <= quality["fermentation_index"] <= 1.0
    assert quality["defect_rate"] >= 0.0
    assert 0.0 <= quality["fine_flavor_probability"] <= 1.0
    assert "price_premium_usd_per_t" in quality


def test_validation_invalid_latitude(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "farm_location": {"lat": 95.0, "lon": -1.2}}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_validation_negative_farm_size(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "farm_size_ha": -1.0}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_validation_missing_cocoa_price_uses_icco(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "cocoa_price_usd"}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 200
    assert response.json()["financial_impact"]["usd"]["price_usd_per_tonne"] > 0


def test_validation_unknown_intervention(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "intervention_type": "unknown_intervention"}
    response = client.post("/simulate-intervention", json=payload)
    assert response.status_code == 422


def test_simulate_with_overridden_model(client: TestClient) -> None:
    app.state.yield_model = YieldSurrogateModel(
        sequence_length=365,
        climate_features=11,
        static_features=SITE_STATIC_DIM,
        galileo_dim=0,
    )
    response = client.post("/simulate-intervention", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "avoided_loss_tonnes" in response.json()


def test_yield_surrogate_galileo_dim_backward_compat() -> None:
    model = YieldSurrogateModel(static_features=SITE_STATIC_DIM, galileo_dim=0)
    assert model.static_features == SITE_STATIC_DIM
    model_g = YieldSurrogateModel(static_features=SITE_STATIC_DIM, galileo_dim=32)
    assert model_g.static_features == SITE_STATIC_DIM + 32
    climate = torch.randn(2, 365, 11)
    static = torch.randn(2, SITE_STATIC_DIM + 32)
    out = model_g(climate, static)
    assert out.shape == (2,)
