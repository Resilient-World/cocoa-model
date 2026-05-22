"""Tests for FDP 2025a default threshold and validation."""

from __future__ import annotations

import pytest

from data.cocoa_exposure import (
    DEFAULT_THRESHOLD,
    MIN_THRESHOLD,
    CocoaExposureIngest,
    validate_threshold,
)


def test_default_threshold_is_fdp_f1_optimal() -> None:
    assert pytest.approx(0.96) == DEFAULT_THRESHOLD


def test_validate_threshold_accepts_default() -> None:
    assert validate_threshold(0.96) == pytest.approx(0.96)


def test_validate_threshold_rejects_below_half() -> None:
    with pytest.raises(ValueError, match=f">= {MIN_THRESHOLD}"):
        validate_threshold(0.49)


def test_ingest_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match=f">= {MIN_THRESHOLD}"):
        CocoaExposureIngest(aoi=object(), threshold=0.2)  # type: ignore[arg-type]


def test_ingest_default_threshold_on_instance() -> None:
    ing = CocoaExposureIngest(aoi=object())  # type: ignore[arg-type]
    assert ing.threshold == pytest.approx(0.96)
