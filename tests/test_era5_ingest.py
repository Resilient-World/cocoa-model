"""Unit tests for ERA5 ingest helpers (no Earth Engine calls)."""

from data.era5_ingest import GHANA_CI_BOUNDS, HEAT_STRESS_THRESHOLD_C, KELVIN_OFFSET


def test_ghana_ci_bounds_cover_both_countries() -> None:
    assert GHANA_CI_BOUNDS["west"] < -8.0
    assert GHANA_CI_BOUNDS["east"] > 0.0
    assert GHANA_CI_BOUNDS["south"] < 5.0
    assert GHANA_CI_BOUNDS["north"] > 10.0


def test_heat_stress_threshold_kelvin() -> None:
    assert HEAT_STRESS_THRESHOLD_C + KELVIN_OFFSET == 305.15
