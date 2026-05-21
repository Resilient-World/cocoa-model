"""Tests for ensemble v2 exposure weights and blending."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.cocoa_exposure import CocoaExposureIngest, sample_cocoa_probability_at_point
from data.ensemble_weights import (
    load_ensemble_weights,
    save_ensemble_weights_yaml,
    validate_weights_sum,
)


def test_ensemble_weights_sum_to_one(tmp_path: Path) -> None:
    doc = {
        "schema_version": 1,
        "default": {"aef": 0.4, "galileo": 0.25, "agrifm": 0.25, "fdp": 0.10},
        "global": {"aef": 0.45, "galileo": 0.30, "agrifm": 0.25},
        "regions": {
            "ghana": {
                "weights": {"aef": 0.35, "galileo": 0.25, "agrifm": 0.30, "fdp": 0.10},
            },
        },
    }
    path = tmp_path / "ensemble_weights.yaml"
    save_ensemble_weights_yaml(doc, path)
    w = load_ensemble_weights("ghana", path=path)
    assert validate_weights_sum(w)
    assert abs(sum(w.values()) - 1.0) < 1e-5


def test_ensemble_v2_blend_probability_range(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        year=2023,
        backend="ensemble_v2",
        region="ghana",
    )
    monkeypatch.setattr(ing, "_aef_probability_at_point", lambda lat, lon: 0.2)
    monkeypatch.setattr(ing, "_galileo_probability_at_point", lambda lat, lon: 0.4)
    monkeypatch.setattr(ing, "_agrifm_probability_at_point", lambda lat, lon: 0.6)
    monkeypatch.setattr(ing, "_fdp_probability_at_point", lambda lat, lon, scale_m: 0.8)
    p = ing._ensemble_v2_blend(6.0, -4.0, scale_m=10)
    assert 0.0 <= p <= 1.0


def test_sample_cocoa_probability_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "data.cocoa_exposure.is_fdp_covered",
        lambda lat, lon: True,
    )
    monkeypatch.setattr(
        "data.cocoa_exposure.initialize_earth_engine",
        lambda project=None: None,
    )

    class _FakeIng:
        def sample_point(self, lat, lon, scale_m=10):
            return 0.42

    monkeypatch.setattr(
        "data.cocoa_exposure.CocoaExposureIngest",
        lambda *a, **k: _FakeIng(),
    )
    p = sample_cocoa_probability_at_point(6.0, -4.0, backend="ensemble_v2")
    assert 0.0 <= p <= 1.0
