"""Tests for online conformal on POST /simulate-scenario."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr
from fastapi.testclient import TestClient

from api.config import APISettings
from api.main import app
from api.online_conformal_store import OnlineConformalStore, stratum_key
from api.scenario_conformal import apply_scenario_conformal, resolve_region
from models.casej_surrogate import CASEJSurrogate
from models.cqr import QuantileYieldSurrogate
from models.eci import ECIIntegral
from models.yield_surrogate import N_CLIMATE_CHANNELS
from tests.conformal_online_helpers import (
    post_shift_coverage,
    run_online_coverage,
)

SCENARIO_PAYLOAD = {
    "farm_location": {"lat": 6.5, "lon": -1.2},
    "farm_size_ha": 5.0,
    "current_yield": 2.0,
    "intervention_type": "shade_trees",
    "cocoa_price_usd": 3200.0,
    "scenario": "ssp245",
    "horizon_year": 2050,
}

SITE_STATIC_DIM = 13
SEQUENCE_LENGTH = 365


class StubFeatureResolver:
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


def _scenario_grid_dataset() -> xr.Dataset:
    times = xr.date_range("2023-01-01", periods=365, freq="D")
    t = len(times)
    shape = (t, 1, 1)
    rng = np.random.default_rng(7)
    tmax = (30.0 + 0.01 * np.arange(t)).reshape(shape).astype(np.float32)
    tmin = (tmax - 7.0).astype(np.float32)
    tmean = (0.5 * (tmax + tmin)).astype(np.float32)
    precip = np.abs(rng.normal(3.0, 1.0, shape)).astype(np.float32)
    return xr.Dataset(
        {
            "tmax": (("time", "latitude", "longitude"), tmax),
            "tmin": (("time", "latitude", "longitude"), tmin),
            "tmean": (("time", "latitude", "longitude"), tmean),
            "precip": (("time", "latitude", "longitude"), precip),
            "rh_mean": (("time", "latitude", "longitude"), np.full(shape, 75.0, dtype=np.float32)),
            "vpd": (("time", "latitude", "longitude"), np.full(shape, 1.0, dtype=np.float32)),
            "et0": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype=np.float32)),
            "cwd": (("time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32)),
            "srad": (("time", "latitude", "longitude"), np.full(shape, 12.0, dtype=np.float32)),
            "wind10m": (("time", "latitude", "longitude"), np.full(shape, 2.0, dtype=np.float32)),
            "sm_root": (("time", "latitude", "longitude"), np.full(shape, 0.3, dtype=np.float32)),
        },
        coords={
            "time": times,
            "latitude": [6.5],
            "longitude": [-1.2],
        },
    )


def _mock_casej_model() -> MagicMock:
    mock = MagicMock(spec=CASEJSurrogate)
    mock.training = False
    mock.eval.return_value = mock
    mock.side_effect = lambda *a, **k: torch.tensor([2.1])
    return mock


def _mock_cqr_model() -> MagicMock:
    model = MagicMock(spec=QuantileYieldSurrogate)
    model.eval.return_value = model

    def forward(climate: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        del static
        base = float(climate.mean())
        return torch.tensor([[base - 0.2, base, base + 0.2]], dtype=torch.float32)

    model.side_effect = forward
    model.__call__ = forward
    return model


@pytest.fixture
def scenario_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with _client_for_method(tmp_path, monkeypatch, "eci_integral") as client:
        yield client


def test_state_persistence_across_reloads(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    initial_path = tmp_path / "init.json"
    store1 = OnlineConformalStore(
        state_path=state_path,
        initial_state_path=initial_path,
        conformal_method="eci_integral",
        eci_eta=2.5,
    )
    key = stratum_key("ssp245", 2030, "ghana")
    updater = store1.get_updater(key)
    updater.update(0.5, covered=True)
    store1.save_after_update(key, updater, covered=True)

    store2 = OnlineConformalStore(
        state_path=state_path,
        initial_state_path=initial_path,
        conformal_method="eci_integral",
    )
    store2.reload_from_disk()
    q_after = store2.get_stratum_state(key).q_t
    assert q_after != 0.0
    assert store2.coverage_running_avg(key) == 1.0


def test_10k_shift_coverage_gate() -> None:
    """ECI-Integral on 10k scores: Wu fixture stream + late distribution shift."""
    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "conformal" / "amazon_prophet_scores.npz"
    )
    base = np.load(fixture)["scores"]
    n = 10_000
    scores = np.tile(base, n // len(base) + 1)[:n].astype(np.float64)
    shift_at = 8000
    scores[shift_at:] += 0.35
    updater = ECIIntegral(0.1, eta=4.0, decay=0.95, window=100, q_init=0.0)
    cov, _, _, qs = run_online_coverage(
        updater,
        scores,
        alpha=0.1,
        burn_in=400,
        warm_start=200,
    )
    assert 0.88 <= cov <= 0.92
    tail_cov = post_shift_coverage(scores, qs, shift_at=shift_at, window=500)
    assert 0.88 <= tail_cov <= 0.92


def _client_for_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str):
    from contextlib import contextmanager

    state_path = tmp_path / f"state_{method}.json"
    initial_path = tmp_path / "initial.json"
    initial_path.write_text("{}", encoding="utf-8")
    hist = tmp_path / "era5_stub"
    cmip = tmp_path / "cmip6_stub"
    hist.mkdir()
    cmip.mkdir()
    monkeypatch.setenv("CONFORMAL_METHOD", method)
    monkeypatch.setenv("ONLINE_CONFORMAL_STATE_PATH", str(state_path))
    monkeypatch.setenv("CONFORMAL_INITIAL_STATE_PATH", str(initial_path))
    monkeypatch.setenv("USE_REAL_FEATURES", "false")

    mock_casej = _mock_casej_model()

    cal = None
    if method == "split_cqr":
        cal = MagicMock()
        cal.empirical_coverage = 0.91
        iv = MagicMock(lower=1.0, upper=2.5, median=1.8)
        cal.predict_interval.return_value = iv

    @contextmanager
    def _cm():
        sample_tensor = torch.full((50,), 2.1)
        with (
            patch("api.main.load_yield_model", return_value=MagicMock()),
            patch("api.main.load_casej_model", return_value=mock_casej),
            patch("api.main.load_conformal_if_exists", return_value=None),
            patch("api.main.load_cqr_bundle", return_value=(_mock_cqr_model(), cal)),
            patch("api.simulation.ScenarioBuilder") as mock_builder,
            patch(
                "api.simulation.predict_scenario_yield_samples",
                return_value=sample_tensor,
            ),
        ):
            inst = mock_builder.return_value
            inst.build_scenario.return_value = _scenario_grid_dataset()
            with TestClient(app) as client:
                client.app.state.settings.era5_zarr_path = hist
                client.app.state.settings.cmip6_zarr_path = cmip
                client.app.state.feature_resolver = StubFeatureResolver()
                client.app.state.casej_model = mock_casej
                yield client

    return _cm()


@pytest.mark.parametrize(
    "conformal_method,expected_ci_method",
    [
        ("eci_integral", "eci_integral"),
        ("aci", "aci"),
        ("split_cqr", "cqr"),
    ],
)
def test_backward_compat_conformal_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conformal_method: str,
    expected_ci_method: str,
) -> None:
    with _client_for_method(tmp_path, monkeypatch, conformal_method) as client:
        r = client.post("/simulate-scenario", json=SCENARIO_PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert "baseline_yield_tonnes_per_ha" in body
    assert body["baseline_yield_tonnes_per_ha"]["mean"] is not None
    ci = body.get("confidence_interval")
    if ci is not None:
        assert ci["method"] == expected_ci_method


def test_online_update_latency_under_5ms() -> None:
    """Online conformal increment (mocked CQR, no disk I/O) stays within +5 ms p95."""
    from api.schemas import SimulateScenarioRequest

    request = SimulateScenarioRequest.model_validate(SCENARIO_PAYLOAD)
    climate = torch.randn(1, 365, N_CLIMATE_CHANNELS)
    static = torch.randn(1, SITE_STATIC_DIM)
    settings = APISettings(conformal_method="aci")
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    store = OnlineConformalStore(state_path=tmp / "lat_bench.json", aci_eta=0.005)
    model = _mock_cqr_model()

    def run_once() -> None:
        apply_scenario_conformal(
            request,
            cqr_model=model,
            cqr_calibrator=None,
            store=store,
            settings=settings,
            climate_baseline=climate,
            climate_projected=climate,
            static_cf=static,
            static_factual=static,
            biotic_cf_frac=1.0,
            biotic_fact_frac=1.0,
        )

    with patch.object(store, "_redis_set_blob"):
        for _ in range(20):
            store._updater_cache.clear()
            run_once()
        times = []
        for _ in range(200):
            store._updater_cache.clear()
            t0 = time.perf_counter()
            run_once()
            times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    assert p95 <= 5.0, f"p95 online conformal overhead {p95:.2f}ms > 5ms"


def test_resolve_region_ghana() -> None:
    assert resolve_region(6.5, -1.2) == "ghana"
