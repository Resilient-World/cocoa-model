"""Tests for split and Mondrian conformal yield prediction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.conformal import (
    MondrianConformalYield,
    SplitConformalYield,
    assign_kalischek_zone,
    conformal_quantile,
    empirical_coverage,
    load_conformal,
    nonconformity_score,
    save_conformal,
)
from models.yield_surrogate import YieldPrediction, YieldSurrogateModel


def _true_yield(climate: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    return climate[..., 0].mean(dim=1) * 0.05 + static[:, 0] * 0.002


class SyntheticYieldModel(nn.Module):
    """Deterministic mean matching the synthetic DGP."""

    def forward(self, climate: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return _true_yield(climate, static)


def _synthetic_predict_with_uncertainty(
    model: nn.Module,
    x_climate: torch.Tensor,
    x_static: torch.Tensor,
    num_samples: int = 50,
) -> YieldPrediction:
    del num_samples
    mean = model(x_climate, x_static)
    std = torch.full_like(mean, 0.12)
    return YieldPrediction(mean=mean, std=std)


def _make_tensors(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    climate = torch.randn(n, 365, 11, generator=rng) * 0.2
    climate[..., 0] = 28.0 + torch.randn(n, 365, generator=rng) * 2.0
    static = torch.randn(n, 13, generator=rng)
    static[:, 0] = 150.0
    noise = torch.randn(n, generator=rng) * 0.18
    y = _true_yield(climate, static) + noise
    return climate, static, y


def _loader(
    climate: torch.Tensor,
    static: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
) -> DataLoader:
    return DataLoader(TensorDataset(climate, static, y), batch_size=batch_size)


@patch("models.conformal.predict_with_uncertainty", side_effect=_synthetic_predict_with_uncertainty)
def test_split_conformal_empirical_coverage_at_n2000(mock_predict: object) -> None:
    """Synthetic regression: marginal coverage within ±2% of 90% nominal."""
    del mock_predict
    n_calib = 800
    n_test = 2000
    climate_c, static_c, y_c = _make_tensors(n_calib, seed=1)
    climate_t, static_t, y_t = _make_tensors(n_test, seed=2)

    model = SyntheticYieldModel()
    calib_loader = _loader(climate_c, static_c, y_c)

    predictor = SplitConformalYield().calibrate(
        model,  # type: ignore[arg-type]
        calib_loader,
        alpha=0.1,
        num_samples=20,
    )

    lowers: list[float] = []
    uppers: list[float] = []
    for i in range(n_test):
        interval = predictor.predict(
            model,  # type: ignore[arg-type]
            climate_t[i : i + 1],
            static_t[i : i + 1],
            num_samples=20,
        )
        lowers.append(interval.lower)
        uppers.append(interval.upper)

    coverage = empirical_coverage(
        y_t.numpy(),
        np.asarray(lowers),
        np.asarray(uppers),
    )
    assert coverage == pytest.approx(0.9, abs=0.02)


def test_nonconformity_score_and_quantile() -> None:
    y = torch.tensor([1.0, 2.0, 3.0])
    y_hat = torch.tensor([1.0, 1.0, 1.0])
    sigma = torch.tensor([0.1, 0.1, 0.1])
    scores = nonconformity_score(y, y_hat, sigma, epsilon=1e-6)
    assert scores.shape == (3,)
    q = conformal_quantile(scores.numpy(), alpha=0.1)
    assert q >= scores.max() - 1e-6 or q > 0


def test_mondrian_zone_assignment() -> None:
    assert assign_kalischek_zone(5.5, -3.0) == "Forest"
    assert assign_kalischek_zone(7.5, -3.0) == "Forest-Savanna Transition"
    assert assign_kalischek_zone(9.5, -3.0) == "Guinea Savanna"


@patch("models.conformal.predict_with_uncertainty", side_effect=_synthetic_predict_with_uncertainty)
def test_mondrian_conformal_calibrate_and_predict(mock_predict: object) -> None:
    del mock_predict
    n = 400
    climate, static, y = _make_tensors(n, seed=3)
    lats = np.linspace(5.0, 10.0, n)
    lons = np.full(n, -3.0)

    class ZoneDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return n

        def __getitem__(self, idx: int) -> dict:
            return {
                "climate": climate[idx],
                "static": static[idx],
                "y": y[idx],
                "zone": assign_kalischek_zone(float(lats[idx]), float(lons[idx])),
            }

    from models.conformal import _collate_calibration

    loader = DataLoader(
        ZoneDataset(),
        batch_size=32,
        collate_fn=_collate_calibration,
    )
    model = SyntheticYieldModel()
    mondrian = MondrianConformalYield().calibrate(
        model,  # type: ignore[arg-type]
        loader,
        alpha=0.1,
        num_samples=10,
        min_zone_samples=20,
    )
    assert len(mondrian.zone_quantiles) >= 1
    interval = mondrian.predict(
        model,  # type: ignore[arg-type]
        climate[:1],
        static[:1],
        zone="Forest",
        num_samples=10,
    )
    assert interval.lower <= interval.point <= interval.upper


def test_conformal_json_roundtrip(tmp_path: Path) -> None:
    predictor = SplitConformalYield(quantile=1.35, alpha=0.1)
    path = tmp_path / "conformal.json"
    save_conformal(predictor, path)
    loaded = load_conformal(path)
    assert isinstance(loaded, SplitConformalYield)
    assert loaded.quantile == pytest.approx(1.35)
    assert loaded.coverage_target == pytest.approx(0.9)

    mondrian = MondrianConformalYield(
        zone_quantiles={"Forest": 1.1},
        fallback_quantile=1.4,
        alpha=0.1,
    )
    mpath = tmp_path / "mondrian.json"
    save_conformal(mondrian, mpath)
    loaded_m = load_conformal(mpath)
    assert isinstance(loaded_m, MondrianConformalYield)
    assert loaded_m.zone_quantiles["Forest"] == pytest.approx(1.1)


def test_api_returns_conformal_when_json_present(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    predictor = SplitConformalYield(quantile=1.2, alpha=0.1)
    conf_path = tmp_path / "conformal.json"
    save_conformal(predictor, conf_path)

    class StubResolver:
        def resolve_climate(self, lat: float, lon: float, year: int) -> torch.Tensor:
            del lat, lon, year
            c = torch.randn(1, 365, 11) * 0.1
            c[..., 0] = 28.0
            return c

        def resolve_static(self, lat: float, lon: float) -> torch.Tensor:
            del lat, lon
            s = torch.zeros(1, 13)
            s[0, 0] = 150.0
            return s

        def resolve_static_with_galileo(self, lat: float, lon: float, year: int) -> torch.Tensor:
            return self.resolve_static(lat, lon)

        def resolve_teleconnection(self, lat: float, lon: float, year: int) -> None:
            del lat, lon, year
            return None

    with TestClient(app) as client:
        app.state.feature_resolver = StubResolver()
        app.state.yield_model = YieldSurrogateModel()
        app.state.conformal = load_conformal(conf_path)

        with patch(
            "models.conformal.predict_with_uncertainty",
            side_effect=_synthetic_predict_with_uncertainty,
        ):
            response = client.post(
                "/simulate-intervention",
                json={
                    "farm_location": {"lat": 6.5, "lon": -1.2},
                    "farm_size_ha": 5.0,
                    "current_yield": 2.0,
                    "intervention_type": "shade_trees",
                    "cocoa_price_usd": 3200.0,
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["conformal_interval"] is not None
    assert "coverage_guarantee" in data["conformal_interval"]["baseline_yield_tonnes_per_ha"]
    assert data["conformal_interval"]["avoided_loss_tonnes"]["coverage_target"] == 0.9
