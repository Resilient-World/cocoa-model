"""Pandera schema tests."""

from __future__ import annotations

import pandas as pd
import pytest

from data.schemas import (
    ERA5DailySchema,
    FDPProbabilitySchema,
    FarmPanelSchema,
    validate_dataframe,
)


def test_era5_valid() -> None:
    df = pd.DataFrame(
        {
            "lat": [6.0],
            "lon": [-2.0],
            "date": [pd.Timestamp("2020-01-01")],
            "tmax": [30.0],
            "tmin": [22.0],
            "precip": [3.0],
            "srad": [15.0],
            "wind10m": [2.0],
        }
    )
    validate_dataframe(ERA5DailySchema, df)


def test_era5_invalid_tmax() -> None:
    df = pd.DataFrame(
        {
            "lat": [6.0],
            "lon": [-2.0],
            "date": [pd.Timestamp("2020-01-01")],
            "tmax": [20.0],
            "tmin": [25.0],
            "precip": [3.0],
            "srad": [15.0],
            "wind10m": [2.0],
        }
    )
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(ERA5DailySchema, df)


def test_fdp_probability() -> None:
    df = pd.DataFrame({"lon": [-3.0], "lat": [6.0], "probability": [0.5], "year": [2023]})
    validate_dataframe(FDPProbabilitySchema, df)


def test_farm_panel() -> None:
    df = pd.DataFrame(
        {
            "farm_id": ["f1"],
            "treatment": [1],
            "yield_pre": [1.2],
            "yield_post": [1.5],
            "farm_size_ha": [5.0],
            "lat": [6.0],
            "lon": [-2.0],
            "cocoa_price_usd": [3200.0],
        }
    )
    validate_dataframe(FarmPanelSchema, df)
