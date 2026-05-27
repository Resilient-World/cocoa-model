"""Unit tests for HybridYieldSurrogate blend logic and fallback behavior."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from models.process.casej_surrogate import CASEJSurrogate
from models.tsfm.ensemble import TsfmEnsemble
from models.tsfm.hybrid_surrogate import HybridYieldSurrogate


@pytest.fixture
def casej_model() -> CASEJSurrogate:
    model = CASEJSurrogate()
    model.eval()
    return model


@pytest.fixture
def mock_ensemble() -> TsfmEnsemble:
    ensemble = MagicMock(spec=TsfmEnsemble)
    from models.tsfm.wrappers import TsfmForecast
    ensemble.forecast.return_value = TsfmForecast(
        p10=np.array([0.4] * 12),
        p50=np.array([0.5] * 12),
        p90=np.array([0.6] * 12),
    )
    return ensemble


class TestHybridSurrogate:
    def test_co2_in_envelope(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        assert hybrid._co2_in_envelope(420.0) is True
        assert hybrid._co2_in_envelope(600.0) is True
        assert hybrid._co2_in_envelope(300.0) is False
        assert hybrid._co2_in_envelope(1000.0) is False

    def test_should_use_tsfm_disabled(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = False
        assert hybrid._should_use_tsfm(420.0) is False

    def test_should_use_tsfm_outside_envelope(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = True
        assert hybrid._should_use_tsfm(1000.0) is False

    def test_should_use_tsfm_enabled_in_envelope(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = True
        assert hybrid._should_use_tsfm(420.0) is True

    def test_forward_casej_only(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = False
        climate = torch.randn(1, 365, 11)
        static = torch.randn(1, 13)
        co2 = torch.tensor([420.0])
        result = hybrid.forward(climate, static, co2_ppm=co2)
        assert result["tsfm_active"] is False
        assert result["blend_weights"]["tsfm"] == 0.0
        assert result["blend_weights"]["casej"] == 1.0
        assert result["blended_mean"] == result["casej_mean"]

    def test_forward_with_tsfm(self, casej_model: CASEJSurrogate, mock_ensemble: TsfmEnsemble) -> None:
        hybrid = HybridYieldSurrogate(casej_model, tsfm_ensemble=mock_ensemble)
        hybrid._enabled = True
        climate = torch.randn(1, 365, 11)
        static = torch.randn(1, 13)
        co2 = torch.tensor([420.0])
        monthly_history = np.random.randn(24, 6).astype(np.float32)
        result = hybrid.forward(
            climate, static, co2_ppm=co2, monthly_history=monthly_history, horizon=12
        )
        assert result["tsfm_active"] is True
        assert result["blend_weights"]["tsfm"] == pytest.approx(0.6)
        assert result["blend_weights"]["casej"] == pytest.approx(0.4)
        assert result["tsfm_mean"] is not None

    def test_forward_tsfm_fallback_on_error(
        self, casej_model: CASEJSurrogate, mock_ensemble: TsfmEnsemble
    ) -> None:
        mock_ensemble.forecast.side_effect = RuntimeError("model error")
        hybrid = HybridYieldSurrogate(casej_model, tsfm_ensemble=mock_ensemble)
        hybrid._enabled = True
        climate = torch.randn(1, 365, 11)
        static = torch.randn(1, 13)
        co2 = torch.tensor([420.0])
        monthly_history = np.random.randn(24, 6).astype(np.float32)
        result = hybrid.forward(
            climate, static, co2_ppm=co2, monthly_history=monthly_history
        )
        assert result["tsfm_active"] is False
        assert result["blend_weights"]["casej"] == 1.0

    def test_forward_no_monthly_history(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = True
        climate = torch.randn(1, 365, 11)
        static = torch.randn(1, 13)
        co2 = torch.tensor([420.0])
        result = hybrid.forward(climate, static, co2_ppm=co2)
        assert result["tsfm_active"] is False

    def test_predict_scenario_alias(self, casej_model: CASEJSurrogate) -> None:
        hybrid = HybridYieldSurrogate(casej_model)
        hybrid._enabled = False
        climate = torch.randn(1, 365, 11)
        static = torch.randn(1, 13)
        co2 = torch.tensor([420.0])
        result = hybrid.predict_scenario(climate, static, co2)
        assert "blended_mean" in result
