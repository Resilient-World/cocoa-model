"""Tests for CASE2 / RCASE2 runner (rpy2 lazy import)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.case2_runner import (
    CASE2NotInstalled,
    CASE2Runner,
    REQUIRED_WEATHER_COLUMNS,
    _prepare_weather,
    _validate_climate,
)


def _toy_weather(n: int = 400) -> pd.DataFrame:
    t = pd.date_range("2010-01-01", periods=n, freq="D")
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "date": t,
            "tmin_c": 22 + rng.normal(0, 1, n),
            "tmax_c": 30 + rng.normal(0, 1, n),
            "precip_mm": np.maximum(0, rng.gamma(1, 5, n)),
            "srad_mj": 18 + rng.normal(0, 1, n),
            "vapor_pressure_kpa": 2.0 + rng.normal(0, 0.1, n),
        }
    )


def test_case2_runner_imports() -> None:
    from models.case2_runner import CASE2Result  # noqa: F401

    assert CASE2Runner is not None


def test_prepare_weather_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        _prepare_weather(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3)}))


def test_validate_climate_temperature_bounds() -> None:
    weather = _toy_weather(400)
    flags = _validate_climate(weather)
    assert flags["mean_temperature_in_bounds"] is True

    bad = weather.copy()
    bad["tmin_c"] = -30.0
    bad["tmax_c"] = -20.0
    with pytest.raises(ValueError, match="Mean temperature"):
        _validate_climate(bad)


def test_management_validation_via_simulate_signature() -> None:
    """CASE2Runner.simulate validates management without requiring RCASE2."""
    weather = _toy_weather(8 * 365)
    soil = {"layer_depths_cm": [50, 50, 50]}
    management = {"tree_age_years": 2, "planting_density": 1100, "slai": 1.0}

    with pytest.raises(CASE2NotInstalled):
        runner = CASE2Runner()
        runner.simulate(weather, soil, management, n_years=8)


def test_required_weather_column_count() -> None:
    assert len(REQUIRED_WEATHER_COLUMNS) == 6
