"""Unit tests for /simulate-scenario corrdiff downscaling (no FastAPI lifespan)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from api.schemas import SimulateScenarioRequest
from api.simulation import simulate_scenario
from counterfactual.corrdiff_downscaler import corrdiff_cache_path, write_synthetic_corrdiff_cache
from models.casej_surrogate import CASEJSurrogate
from tests.test_api_scenario import SCENARIO_PAYLOAD, StubFeatureResolver


@pytest.fixture
def scenario_dirs(tmp_path: Path) -> tuple[Path, Path]:
    hist = tmp_path / "era5_stub"
    cmip = tmp_path / "cmip6_stub"
    hist.mkdir()
    cmip.mkdir()
    return hist, cmip


def test_simulate_scenario_corrdiff_from_cache(
    scenario_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    hist, cmip = scenario_dirs
    cache = corrdiff_cache_path(tmp_path, "ssp245", 2050, "ghana")
    write_synthetic_corrdiff_cache(
        cache,
        scenario="ssp245",
        horizon=2050,
        region="ghana",
        n_samples=2,
    )
    settings = MagicMock()
    settings.scenario_yield_backend = "casej"
    settings.corrdiff_processed_dir = tmp_path
    settings.corrdiff_allow_inline = False
    settings.yield_blend_weight = 0.0
    settings.drift_enabled = False

    request = SimulateScenarioRequest.model_validate(
        {**SCENARIO_PAYLOAD, "downscaling_method": "corrdiff"}
    )
    model = CASEJSurrogate(
        sequence_length=365, climate_features=11, static_features=13, galileo_dim=0
    )
    resolver = StubFeatureResolver()

    with patch("api.simulation.predict_scenario_yield_samples") as mock_pred:
        mock_pred.return_value = torch.tensor([2.0, 2.1, 2.2])
        resp = simulate_scenario(
            request,
            model,
            resolver,
            historical_zarr_path=hist,
            cmip6_zarr_path=cmip,
            num_samples=3,
            settings=settings,
        )

    assert resp.downscaling_method == "corrdiff"
    assert resp.corrdiff_samples_used == 2
    assert mock_pred.call_count == 4  # 2 samples × (baseline + projected)


def test_simulate_scenario_corrdiff_cache_miss_raises(
    scenario_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    hist, cmip = scenario_dirs
    settings = MagicMock()
    settings.corrdiff_processed_dir = tmp_path / "empty"
    settings.corrdiff_processed_dir.mkdir()
    settings.corrdiff_allow_inline = False

    request = SimulateScenarioRequest.model_validate(
        {**SCENARIO_PAYLOAD, "downscaling_method": "corrdiff"}
    )
    model = CASEJSurrogate(
        sequence_length=365, climate_features=11, static_features=13, galileo_dim=0
    )

    with pytest.raises(ValueError, match="CorrDiff cache"):
        simulate_scenario(
            request,
            model,
            StubFeatureResolver(),
            historical_zarr_path=hist,
            cmip6_zarr_path=cmip,
            settings=settings,
        )


@patch("api.simulation.ScenarioBuilder")
def test_default_request_still_linear(
    mock_sb_cls: MagicMock, scenario_dirs: tuple[Path, Path]
) -> None:
    from tests.test_api_scenario import _scenario_grid_dataset

    hist, cmip = scenario_dirs
    inst = mock_sb_cls.return_value
    inst.build_scenario.return_value = _scenario_grid_dataset()

    request = SimulateScenarioRequest.model_validate(SCENARIO_PAYLOAD)
    model = CASEJSurrogate(
        sequence_length=365, climate_features=11, static_features=13, galileo_dim=0
    )
    settings = MagicMock()
    settings.scenario_yield_backend = "casej"
    settings.yield_blend_weight = 0.0
    settings.drift_enabled = False

    with patch("api.simulation.predict_scenario_yield_samples") as mock_pred:
        mock_pred.return_value = torch.tensor([2.0, 2.1, 2.2])
        resp = simulate_scenario(
            request,
            model,
            StubFeatureResolver(),
            historical_zarr_path=hist,
            cmip6_zarr_path=cmip,
            settings=settings,
        )

    assert resp.downscaling_method == "linear_delta"
    assert resp.corrdiff_samples_used is None
