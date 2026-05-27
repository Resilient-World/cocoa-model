"""Unit tests for TSFM wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.tsfm.wrappers import (
    Chronos2Wrapper,
    Moirai2Wrapper,
    TimeMoEWrapper,
    TimesFM2Wrapper,
    TsfmForecast,
    build_wrapper,
)


class TestTsfmForecast:
    def test_valid_forecast(self) -> None:
        fc = TsfmForecast(
            p10=np.array([0.3, 0.4]),
            p50=np.array([0.5, 0.6]),
            p90=np.array([0.7, 0.8]),
        )
        assert fc.p10.shape == (2,)
        assert fc.p50.shape == (2,)
        assert fc.p90.shape == (2,)

    def test_scalar_raises(self) -> None:
        with pytest.raises(ValueError):
            TsfmForecast(p10=np.float64(0.5), p50=np.float64(0.6), p90=np.float64(0.7))


class TestBuildWrapper:
    def test_valid_names(self) -> None:
        for name in ("chronos-2", "timesfm", "timemoe", "moirai"):
            wrapper = build_wrapper(name, device="cpu")
            assert wrapper.model_id is not None

    def test_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown TSFM model"):
            build_wrapper("nonexistent")


class TestChronos2Wrapper:
    def test_extract_target(self) -> None:
        w = Chronos2Wrapper(device="cpu")
        hist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = w._extract_target(hist)
        np.testing.assert_array_equal(target, np.array([1.0, 3.0], dtype=np.float32))

    def test_extract_target_1d(self) -> None:
        w = Chronos2Wrapper(device="cpu")
        hist = np.array([1.0, 3.0], dtype=np.float32)
        target = w._extract_target(hist)
        np.testing.assert_array_equal(target, np.array([1.0, 3.0], dtype=np.float32))

    def test_compute_quantiles(self) -> None:
        w = Chronos2Wrapper(device="cpu")
        samples = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.9, 1.0, 1.1]])
        p10, p50, p90 = w._compute_quantiles(samples)
        np.testing.assert_allclose(p10, [0.18, 0.28, 0.38], atol=0.01)
        np.testing.assert_allclose(p50, [0.5, 0.6, 0.7], atol=0.01)
        np.testing.assert_allclose(p90, [0.82, 0.92, 1.02], atol=0.01)


class TestTimeMoEWrapper:
    def test_extract_target(self) -> None:
        w = TimeMoEWrapper(device="cpu")
        hist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = w._extract_target(hist)
        np.testing.assert_array_equal(target, np.array([1.0, 3.0], dtype=np.float32))


class TestTimesFM2Wrapper:
    def test_extract_target(self) -> None:
        w = TimesFM2Wrapper(device="cpu")
        hist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = w._extract_target(hist)
        np.testing.assert_array_equal(target, np.array([1.0, 3.0], dtype=np.float32))


class TestMoirai2Wrapper:
    def test_extract_target(self) -> None:
        w = Moirai2Wrapper(device="cpu")
        hist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target = w._extract_target(hist)
        np.testing.assert_array_equal(target, np.array([1.0, 3.0], dtype=np.float32))
