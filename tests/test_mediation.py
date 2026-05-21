"""Tests for causal mediation (NDE/NIE) and intervention API wiring."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from analysis.mediation import mediation_analysis


def _linear_scm_panel(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """M ≈ T; Y = 2*T + M → TE≈3, NDE≈2, NIE≈1 (tiny M noise avoids T–M collinearity)."""
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.5).astype(int)
    m = t.astype(float) + rng.normal(0.0, 0.01, size=n)
    y = 2.0 * t + 1.0 * m
    return pd.DataFrame({"t": t, "m": m, "y": y})


def _linear_nuisance_models(random_state: int):
    del random_state
    return (
        LinearRegression(),
        LogisticRegression(max_iter=500),
    )


def test_mediation_linear_scm_nde_nie() -> None:
    df = _linear_scm_panel(n=2000, seed=1)
    with patch("analysis.mediation.default_nuisance_models", _linear_nuisance_models):
        res = mediation_analysis(
            df,
            treatment_col="t",
            outcome_col="y",
            mediator_col="m",
            covariate_cols=[],
            n_bootstrap=100,
            random_state=42,
        )
    assert abs(res.nde - 2.0) < 0.2
    assert abs(res.nie - 1.0) < 0.2
    assert abs(res.total_effect - 3.0) < 0.25


def test_rho_sensitivity_monotone() -> None:
    df = _linear_scm_panel(n=1500, seed=2)
    with patch("analysis.mediation.default_nuisance_models", _linear_nuisance_models):
        res = mediation_analysis(
            df,
            treatment_col="t",
            outcome_col="y",
            mediator_col="m",
            covariate_cols=[],
            n_bootstrap=50,
            random_state=7,
        )
    assert res.sensitivity_curve
    nies = [row["nie_adjusted"] for row in res.sensitivity_curve]
    assert nies[0] >= nies[-1] - 1e-6


def test_simulate_intervention_mediation_field(tmp_path) -> None:
    from unittest.mock import MagicMock, patch

    import torch
    from api.config import APISettings
    from api.schemas import FarmLocation, InterventionType, SimulateInterventionRequest
    from api.simulation import simulate_intervention
    from models.yield_surrogate import YieldSurrogateModel

    settings = APISettings(
        era5_zarr_path=tmp_path / "era5.zarr",
        use_real_features=False,
        mediation_n_bootstrap=80,
    )
    os.environ["USE_REAL_FEATURES"] = "false"

    req = SimulateInterventionRequest(
        farm_location=FarmLocation(lat=6.12, lon=-5.34),
        farm_size_ha=2.0,
        current_yield=1.5,
        intervention_type=InterventionType.shade_trees,
        decompose_mediators=["microclimate", "soil_moisture", "cssvd_prevalence"],
    )
    model = YieldSurrogateModel()
    resolver = MagicMock()
    resolver.resolve_climate.return_value = torch.randn(1, 365, 11)
    resolver.resolve_static_with_galileo.return_value = torch.randn(1, 13)
    resolver.resolve_teleconnection.return_value = None

    with patch("api.simulation.apply_biotic_losses") as mock_bio:
        mock_bio.return_value = {
            "surviving_fraction": 0.95,
            "total_loss_fraction": 0.05,
            "loss_attribution": {"black_pod": 0.02, "cssvd": 0.02, "mirids": 0.01},
        }
        resp = simulate_intervention(
            req,
            model,
            resolver,
            num_samples=40,
            settings=settings,
        )

    assert resp.mediation is not None
    assert len(resp.mediation.per_mediator) == 3
    mediators = {m.mediator for m in resp.mediation.per_mediator}
    assert mediators == {"microclimate", "soil_moisture", "cssvd_prevalence"}
