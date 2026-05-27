"""Unit tests for TSFM ensemble aggregation and weight fitting."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.tsfm.ensemble import (
    NnlsWeightFitter,
    TsfmEnsemble,
    WeightedMedianForecast,
)
from models.tsfm.wrappers import TsfmForecast


class TestWeightedMedianForecast:
    def test_uniform_weights(self) -> None:
        agg = WeightedMedianForecast()
        forecasts = {
            "timemoe": TsfmForecast(p10=np.array([0.1]), p50=np.array([0.5]), p90=np.array([0.9])),
            "chronos-2": TsfmForecast(p10=np.array([0.2]), p50=np.array([0.6]), p90=np.array([1.0])),
            "timesfm": TsfmForecast(p10=np.array([0.15]), p50=np.array([0.55]), p90=np.array([0.95])),
            "moirai": TsfmForecast(p10=np.array([0.25]), p50=np.array([0.65]), p90=np.array([1.05])),
        }
        result = agg.aggregate(forecasts)
        np.testing.assert_allclose(result.p50, [0.575], atol=0.01)

    def test_custom_weights(self) -> None:
        agg = WeightedMedianForecast(weights={"timemoe": 0.5, "chronos-2": 0.3, "timesfm": 0.2})
        forecasts = {
            "timemoe": TsfmForecast(p10=np.array([0.1]), p50=np.array([0.5]), p90=np.array([0.9])),
            "chronos-2": TsfmForecast(p10=np.array([0.2]), p50=np.array([0.6]), p90=np.array([1.0])),
            "timesfm": TsfmForecast(p10=np.array([0.15]), p50=np.array([0.55]), p90=np.array([0.95])),
        }
        result = agg.aggregate(forecasts)
        expected = 0.5 * 0.5 + 0.3 * 0.6 + 0.2 * 0.55
        np.testing.assert_allclose(result.p50, [expected], atol=0.01)

    def test_empty_forecasts_raises(self) -> None:
        agg = WeightedMedianForecast()
        with pytest.raises(ValueError, match="No active model forecasts"):
            agg.aggregate({})

    def test_multi_horizon(self) -> None:
        agg = WeightedMedianForecast()
        forecasts = {
            "timemoe": TsfmForecast(
                p10=np.array([0.1, 0.2]), p50=np.array([0.5, 0.6]), p90=np.array([0.9, 1.0])
            ),
            "chronos-2": TsfmForecast(
                p10=np.array([0.15, 0.25]), p50=np.array([0.55, 0.65]), p90=np.array([0.95, 1.05])
            ),
        }
        result = agg.aggregate(forecasts)
        assert result.p50.shape == (2,)
        np.testing.assert_allclose(result.p50, [0.525, 0.625], atol=0.01)


class TestNnlsWeightFitter:
    def test_fit_synthetic(self) -> None:
        rng = np.random.default_rng(42)
        region_panels: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for region in ("GHA", "CIV"):
            folds = []
            for _ in range(5):
                hist = rng.normal(0.5, 0.05, 24).astype(np.float32)
                cov = rng.normal(0, 1, (24, 3)).astype(np.float32)
                actual = rng.normal(0.5, 0.05, 12).astype(np.float32)
                folds.append((hist, cov, actual))
            region_panels[region] = folds

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "tsfm_weights.yaml"
            fitter = NnlsWeightFitter(weights_path)

            from unittest.mock import patch
            from models.tsfm.wrappers import TsfmForecast

            def _mock_forecast(history, horizon, num_samples=100):
                return TsfmForecast(
                    p10=np.full(horizon, 0.45, dtype=np.float64),
                    p50=np.full(horizon, 0.5, dtype=np.float64),
                    p90=np.full(horizon, 0.55, dtype=np.float64),
                )

            with patch("models.tsfm.ensemble.build_wrapper") as mock_build:
                mock_wrapper = MagicMock()
                mock_wrapper.forecast = _mock_forecast
                mock_build.return_value = mock_wrapper
                weights = fitter.fit(region_panels, horizon=12, num_samples=10, device="cpu")

            assert "GHA" in weights
            assert "CIV" in weights
            for region_weights in weights.values():
                total = sum(region_weights.values())
                assert abs(total - 1.0) < 0.01


class TestTsfmEnsemble:
    def test_init(self) -> None:
        ensemble = TsfmEnsemble(device="cpu")
        assert ensemble.ensemble_mode == "nnls"
        assert ensemble.max_workers == 4

    def test_get_weights_default(self) -> None:
        ensemble = TsfmEnsemble(device="cpu")
        weights = ensemble._get_weights("unknown_region")
        assert set(weights.keys()) == {"chronos-2", "timesfm", "timemoe", "moirai"}
        assert abs(sum(weights.values()) - 1.0) < 0.01
