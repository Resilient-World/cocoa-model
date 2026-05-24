"""Tests for GEDI/ICESat-2 canopy feature sampling."""

from __future__ import annotations

import pytest

from data.gedi_canopy import GEDICanopyIngest, sample_canopy_at_point


def test_sample_canopy_at_point_mock_fields() -> None:
    sample = sample_canopy_at_point(6.5, -1.2, 2023, use_mock=True)
    assert 0.0 <= sample.canopy_height_m <= 45.0
    assert 0.0 <= sample.canopy_cover_pct <= 100.0
    assert sample.agb_mg_ha >= 0.0
    assert sample.height_uncertainty_m > 0.0
    assert sample.gedi_n_shots >= 0
    assert sample.source_attributions


def test_gedi_canopy_ingest_mock_dataset_bands() -> None:
    ds = GEDICanopyIngest(None, 2023, use_mock=True).build()
    assert {
        "canopy_height_m",
        "canopy_cover_pct",
        "agb_mg_ha",
        "height_uncertainty_m",
        "gedi_n_shots",
    }.issubset(ds.data_vars)


@pytest.mark.integration
def test_sample_canopy_at_point_integration_fallback() -> None:
    sample = sample_canopy_at_point(6.5, -1.2, 2023)
    assert sample.canopy_height_m >= 0.0
