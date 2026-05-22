"""Unit tests for Aurora on /simulate-scenario (light + optional full stack)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_aurora_runner():
    path = _SRC / "counterfactual" / "aurora_runner.py"
    spec = importlib.util.spec_from_file_location("aurora_runner_api_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aurora_runner_api_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_stratum_key():
    path = _SRC / "api" / "online_conformal_store.py"
    spec = importlib.util.spec_from_file_location("online_conformal_store_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["online_conformal_store_test"] = mod
    spec.loader.exec_module(mod)
    return mod.stratum_key


def test_stratum_key_aurora_suffix() -> None:
    stratum_key = _load_stratum_key()
    assert stratum_key("ssp245", 2050, "ghana", downscaling_method="aurora") == "ssp245:2050:ghana:aurora"


def test_aurora_commercial_gate_production() -> None:
    mod = _load_aurora_runner()
    with pytest.raises(ValueError, match="AURORA_COMMERCIAL_OK"):
        mod.check_aurora_commercial_gate(commercial_ok=False, deployment_environment="production")


def test_build_aurora_source_attribution_fields() -> None:
    mod = _load_aurora_runner()
    runner = mod.AuroraScenarioRunner(cache_dir=Path("/tmp"), model_size="small", mock=True)
    attrs = mod.build_aurora_source_attribution(runner)
    assert attrs[0]["aurora_model_version"] == "aurora-0.25-small-pretrained"
    assert attrs[0]["aurora_lora_id"] in ("base", "ghana")


@pytest.mark.integration
def test_simulate_scenario_aurora_mock(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from api.schemas import SimulateScenarioRequest
    from api.simulation import simulate_scenario
    from models.casej_surrogate import CASEJSurrogate
    from tests.test_api_scenario import SCENARIO_PAYLOAD, StubFeatureResolver

    hist = tmp_path / "era5_stub"
    cmip = tmp_path / "cmip6_stub"
    hist.mkdir()
    cmip.mkdir()
    settings = MagicMock()
    settings.aurora_enabled = True
    settings.aurora_commercial_ok = True
    settings.aurora_mock = True
    settings.aurora_model_size = "small"
    settings.aurora_cache_dir = tmp_path / "aurora_cache"
    settings.otel_deployment_environment = "local"
    settings.scenario_yield_backend = "casej"
    settings.yield_blend_weight = 0.0
    settings.drift_enabled = False

    request = SimulateScenarioRequest.model_validate(
        {**SCENARIO_PAYLOAD, "downscaling_method": "aurora"}
    )
    model = CASEJSurrogate(
        sequence_length=365, climate_features=11, static_features=13, galileo_dim=0
    )

    with patch("api.simulation.predict_scenario_yield_samples") as mock_pred:
        mock_pred.return_value = __import__("torch").tensor([2.0, 2.1, 2.2])
        resp = simulate_scenario(
            request,
            model,
            StubFeatureResolver(),
            historical_zarr_path=hist,
            cmip6_zarr_path=cmip,
            num_samples=3,
            settings=settings,
        )

    assert resp.downscaling_method == "aurora"
    assert len(resp.source_attributions) == 1
    assert resp.source_attributions[0].aurora_model_version == "aurora-0.25-small-pretrained"
    assert resp.source_attributions[0].aurora_lora_id in ("base", "ghana")


@pytest.mark.integration
def test_simulate_scenario_aurora_disabled_raises(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from api.schemas import SimulateScenarioRequest
    from api.simulation import simulate_scenario
    from models.casej_surrogate import CASEJSurrogate
    from tests.test_api_scenario import SCENARIO_PAYLOAD, StubFeatureResolver

    hist = tmp_path / "era5_stub"
    cmip = tmp_path / "cmip6_stub"
    hist.mkdir()
    cmip.mkdir()
    settings = MagicMock()
    settings.aurora_enabled = False

    request = SimulateScenarioRequest.model_validate(
        {**SCENARIO_PAYLOAD, "downscaling_method": "aurora"}
    )
    model = CASEJSurrogate(
        sequence_length=365, climate_features=11, static_features=13, galileo_dim=0
    )

    with pytest.raises(ValueError, match="AURORA_ENABLED"):
        simulate_scenario(
            request,
            model,
            StubFeatureResolver(),
            historical_zarr_path=hist,
            cmip6_zarr_path=cmip,
            settings=settings,
        )
