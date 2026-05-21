"""Tests for ensemble v3 weights and blending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

from data.cocoa_exposure import CocoaExposureIngest
from data.ensemble_weights import (
    V3_BACKEND_KEYS,
    load_ensemble_v3_weights,
    save_ensemble_weights_yaml,
    validate_weights_sum,
)


def test_ensemble_v3_weights_sum_to_one(tmp_path: Path) -> None:
    doc = {
        "schema_version": 1,
        "default": {
            "aef": 0.30,
            "galileo": 0.20,
            "agrifm": 0.20,
            "terramind": 0.20,
            "fdp": 0.10,
        },
        "global": {"aef": 0.35, "galileo": 0.25, "agrifm": 0.20, "terramind": 0.20},
        "regions": {
            "ghana": {
                "weights": {
                    "aef": 0.25,
                    "galileo": 0.20,
                    "agrifm": 0.20,
                    "terramind": 0.25,
                    "fdp": 0.10,
                },
            },
        },
    }
    path = tmp_path / "ensemble_weights_v3.yaml"
    save_ensemble_weights_yaml(doc, path)
    w = load_ensemble_v3_weights("ghana", path=path)
    assert validate_weights_sum(w)
    assert abs(sum(w.values()) - 1.0) < 1e-5


def test_nnls_weights_normalize_to_simplex() -> None:
    rng = np.random.default_rng(0)
    n = 40
    matrix = rng.random((n, len(V3_BACKEND_KEYS)))
    labels = rng.random(n)
    coef, _ = nnls(matrix, labels)
    assert coef.sum() > 0
    weights = coef / coef.sum()
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w >= -1e-9 for w in weights)


def test_ensemble_v3_blend_probability_range(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        year=2023,
        backend="ensemble_v3",
        region="ghana",
    )
    monkeypatch.setattr(ing, "_aef_probability_at_point", lambda lat, lon: 0.2)
    monkeypatch.setattr(ing, "_galileo_probability_at_point", lambda lat, lon: 0.3)
    monkeypatch.setattr(ing, "_agrifm_probability_at_point", lambda lat, lon: 0.4)
    monkeypatch.setattr(ing, "_terramind_probability_at_point", lambda lat, lon: 0.5)
    monkeypatch.setattr(ing, "_fdp_probability_at_point", lambda lat, lon, scale_m: 0.6)
    p = ing._ensemble_v3_blend(6.0, -4.0, scale_m=10)
    assert 0.0 <= p <= 1.0
