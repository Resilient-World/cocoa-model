"""Unit tests for Sentinel composite constants (no Earth Engine calls)."""

from data.sentinel_composite import (
    DRY_SEASON_END,
    DRY_SEASON_START,
    GHANA_BOUNDS,
    QA60_CIRRUS_BIT,
    QA60_CLOUD_BIT,
    S2_OPTICAL_BANDS,
)


def test_ghana_bounds() -> None:
    assert GHANA_BOUNDS["west"] < 0
    assert GHANA_BOUNDS["east"] > 0
    assert GHANA_BOUNDS["north"] > 10


def test_dry_season_window() -> None:
    assert DRY_SEASON_START == "2023-12-01"
    assert DRY_SEASON_END == "2024-04-01"


def test_qa60_bit_masks() -> None:
    assert QA60_CLOUD_BIT == 1024
    assert QA60_CIRRUS_BIT == 2048


def test_s2_optical_bands_include_ndvi_inputs() -> None:
    assert "B2" in S2_OPTICAL_BANDS
    assert "B4" in S2_OPTICAL_BANDS
    assert "B8" in S2_OPTICAL_BANDS
