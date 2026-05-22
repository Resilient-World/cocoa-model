"""Tests for Dumont et al. landscape CSSVD incidence model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.cssvd_strain_atlas import lookup_strain_region
from data.dumont_supplement import (
    generate_synthetic_supplement,
    join_exposure_features,
    normalize_dumont_columns,
)
from hazards.composite import apply_biotic_losses
from hazards.cssvd import CSSVDRiskModel
from hazards.cssvd_landscape import (
    HORIZON_MONTHS,
    LandscapeCSSVDModel,
    features_to_dataframe,
    fit_synthetic_demo,
    incidence_probability_at_horizon,
)


def _synthetic_climate():
    import xarray as xr

    n = 60
    time = pd.date_range("2023-01-01", periods=n, freq="D")
    return xr.Dataset(
        {
            "rh_mean": ("time", np.full(n, 78.0, dtype=np.float32)),
            "tmean": ("time", np.full(n, 24.0, dtype=np.float32)),
            "precip": ("time", np.full(n, 6.0, dtype=np.float32)),
        },
        coords={"time": time},
    )


def test_strain_lookup_ghana_coastal():
    region = lookup_strain_region(6.0, -2.0)
    assert region in ("1A", "1B", "1C", "2")


def test_fit_synthetic_demo_produces_predictions():
    model = fit_synthetic_demo(n_samples=200, random_state=0)
    feats = {
        "cocoa_probability_local": 0.8,
        "non_cocoa_buffer_500m": 0.9,
        "canopy_fragmentation_index": 1.5,
        "extreme_precip_5day_count_yr": 5.0,
        "dtr_growing_season": 7.0,
        "strain_1A": 0.0,
        "strain_1B": 0.0,
        "strain_1C": 0.0,
    }
    low = model.predict_from_features(feats)
    feats_high_risk = dict(feats)
    feats_high_risk["non_cocoa_buffer_500m"] = 0.15
    feats_high_risk["extreme_precip_5day_count_yr"] = 35.0
    feats_high_risk["dtr_growing_season"] = 12.0
    high = model.predict_from_features(feats_high_risk)
    assert 0.0 <= low.point <= 1.0
    assert high.point >= low.point
    assert low.pi_high >= low.pi_low


def test_bootstrap_pi_width_positive():
    model = fit_synthetic_demo(n_samples=150, random_state=1)
    feats = {
        "cocoa_probability_local": 0.7,
        "non_cocoa_buffer_500m": 0.5,
        "canopy_fragmentation_index": 1.0,
        "extreme_precip_5day_count_yr": 12.0,
        "dtr_growing_season": 8.0,
        "strain_1A": 1.0,
        "strain_1B": 0.0,
        "strain_1C": 0.0,
    }
    pred = model.predict_from_features(feats)
    assert pred.pi_high - pred.pi_low > 1e-4


def test_save_load_roundtrip(tmp_path: Path):
    model = fit_synthetic_demo(n_samples=120, random_state=3)
    ckpt = tmp_path / "cssvd_landscape.joblib"
    model.save(ckpt)
    loaded = LandscapeCSSVDModel.from_checkpoint(ckpt)
    feats = {
        "cocoa_probability_local": 0.75,
        "non_cocoa_buffer_500m": 0.6,
        "canopy_fragmentation_index": 1.1,
        "extreme_precip_5day_count_yr": 8.0,
        "dtr_growing_season": 9.0,
        "strain_1A": 0.0,
        "strain_1B": 0.0,
        "strain_1C": 0.0,
    }
    p0 = model.predict_from_features(feats).point
    p1 = loaded.predict_from_features(feats).point
    assert p0 == pytest.approx(p1, rel=1e-5)


def test_cssvd_yield_loss_from_precomputed_features(tmp_path: Path):
    model = fit_synthetic_demo(n_samples=100, random_state=4)
    ckpt = tmp_path / "cssvd.joblib"
    model.save(ckpt)

    feats = {
        "cocoa_probability_local": 0.8,
        "non_cocoa_buffer_500m": 0.7,
        "canopy_fragmentation_index": 1.2,
        "extreme_precip_5day_count_yr": 6.0,
        "dtr_growing_season": 7.5,
        "strain_1A": 0.0,
        "strain_1B": 0.0,
        "strain_1C": 0.0,
    }
    risk = CSSVDRiskModel.with_landscape_checkpoint(ckpt)
    inc = risk.landscape_model.predict_from_features(feats)
    loss = risk.annual_yield_loss_fraction(100.0 * inc.point)
    ds = _synthetic_climate()
    out = apply_biotic_losses(
        2.0,
        ds,
        {
            "use_cssvd_landscape": True,
            "cssvd_landscape_checkpoint": str(ckpt),
            "cssvd_landscape_features": feats,
            "lat": 6.0,
            "lon": -5.0,
            "year": 2023,
        },
    )
    assert out["cssvd_landscape"]["cssvd_incidence_prob_12mo"] == pytest.approx(inc.point, rel=1e-4)
    assert 0.0 <= loss <= 0.25


def test_join_exposure_features_synthetic():
    df = generate_synthetic_supplement(n_plots=20, seed=0)
    plots = normalize_dumont_columns(df)
    merged = join_exposure_features(plots, 2023, refresh_cache=True)
    assert "non_cocoa_buffer_500m" in merged.columns
    assert len(merged) == 20
    assert "duration" in merged.columns


def test_incidence_at_horizon_bounds():
    model = fit_synthetic_demo(n_samples=80, random_state=5)
    X = features_to_dataframe(
        [
            {
                "cocoa_probability_local": 0.6,
                "non_cocoa_buffer_500m": 0.5,
                "canopy_fragmentation_index": 1.0,
                "extreme_precip_5day_count_yr": 10.0,
                "dtr_growing_season": 8.0,
                "strain_1A": 0.0,
                "strain_1B": 0.0,
                "strain_1C": 0.0,
            }
        ]
    )
    probs = incidence_probability_at_horizon(model._model, X, horizon_months=HORIZON_MONTHS)  # type: ignore[arg-type]
    assert 0.0 <= probs[0] <= 1.0
