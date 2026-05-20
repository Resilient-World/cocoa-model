"""Tests for ICCO/station yield calibration training."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from api.config import APISettings
from api.simulation import simulate_intervention
from data.yield_panel import build_yield_panel
from models.yield_surrogate import YieldSurrogateModel
from training.train_yield import train_yield_surrogate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_smoke_one_epoch_on_synthetic_panel(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "max_epochs": 1,
            "lr": 3.0e-4,
            "weight_decay": 1.0e-4,
            "batch_size": 16,
            "num_workers": 0,
            "seed": 0,
            "checkpoint_path": str(tmp_path / "yield_smoke.pt"),
            "mlflow_experiment": "test-yield",
            "mlflow_run_name": "smoke",
            "icco_glob": str(_repo_root() / "data/external/icco_*.csv"),
            "crig_path": str(_repo_root() / "data/raw/crig_station_yields.csv"),
            "bootstrap_per_country_year": 2,
            "augment_sigma_t_ha": 0.4,
            "sequence_length": 365,
            "val_fraction": 0.15,
            "device": "cpu",
            "log_every_n_epochs": 1,
        }
    )
    out = train_yield_surrogate(cfg)
    assert out.is_file()
    state = torch.load(out, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)


def test_checkpoint_loads_and_predicts_on_api_input_shape(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "max_epochs": 1,
            "lr": 3.0e-4,
            "weight_decay": 1.0e-4,
            "batch_size": 8,
            "num_workers": 0,
            "seed": 1,
            "checkpoint_path": str(tmp_path / "yield_api_shape.pt"),
            "mlflow_experiment": "test-yield",
            "mlflow_run_name": "shape",
            "icco_glob": str(_repo_root() / "data/external/icco_*.csv"),
            "crig_path": str(_repo_root() / "data/raw/crig_station_yields.csv"),
            "bootstrap_per_country_year": 1,
            "augment_sigma_t_ha": 0.4,
            "sequence_length": 365,
            "val_fraction": 0.15,
            "device": "cpu",
            "log_every_n_epochs": 1,
        }
    )
    ckpt = train_yield_surrogate(cfg)
    model = YieldSurrogateModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)
    model.eval()

    row = build_yield_panel(bootstrap_per_country_year=1, seed=2)[0]
    climate = torch.from_numpy(row.climate).unsqueeze(0)
    static = torch.from_numpy(row.static).unsqueeze(0)
    pred = model(climate, static)
    assert pred.shape == (1,)


def test_blend_weight_zero_path_uses_pure_model(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from api.schemas import FarmLocation, InterventionType, SimulateInterventionRequest

    model = YieldSurrogateModel()
    torch.save(model.state_dict(), tmp_path / "yield.pt")

    class _Resolver:
        def resolve_climate(self, lat: float, lon: float, year: int) -> torch.Tensor:
            return torch.randn(1, 365, 11)

        def resolve_static_with_galileo(self, lat: float, lon: float, year: int) -> torch.Tensor:
            s = torch.zeros(1, 10)
            s[0, 0] = 140.0
            return s

    caplog.set_level(logging.WARNING)
    req = SimulateInterventionRequest(
        farm_location=FarmLocation(lat=6.5, lon=-1.5),
        farm_size_ha=10.0,
        current_yield=0.5,
        intervention_type=InterventionType.shade_trees,
        cocoa_price_usd=3000.0,
    )
    resp = simulate_intervention(
        req,
        model,
        _Resolver(),  # type: ignore[arg-type]
        num_samples=5,
        yield_blend_weight=0.0,
    )
    assert resp.baseline_yield_tonnes_per_ha != req.current_yield
    assert not any("yield_blend_weight" in r.message for r in caplog.records)


def test_api_settings_default_blend_zero() -> None:
    settings = APISettings()
    assert settings.yield_blend_weight == 0.0
    assert settings.model_checkpoint_path == "models/yield_surrogate_v1.pt"
