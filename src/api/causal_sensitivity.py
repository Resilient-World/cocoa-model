"""Cooperative-level DVDS sensitivity bounds for :func:`api.simulation.simulate_intervention`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from analysis.dvds import dvds_ate, tipping_point
from api.schemas import SensitivityBounds
from data.farm_panel import (
    PSM_COVARIATE_COLS,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_real_panel,
    load_synthetic_panel,
    treatment_year_index,
)

if TYPE_CHECKING:
    from api.config import APISettings


def load_dvds_panel(settings: APISettings) -> pd.DataFrame:
    """Load farm panel from parquet or synthetic fallback (CI / local dev)."""
    path = settings.farm_panel_parquet_path
    if path.is_file():
        panel = load_real_panel(path)
    else:
        panel = load_synthetic_panel(n_farms=1000, seed=42)
    if "tmean_annual" not in panel.columns:
        panel = join_climate(panel)
    if "biotic_total_loss_fraction" not in panel.columns:
        panel = join_biotic(panel)
    return panel


def build_dvds_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per farm with yield delta outcome and PSM covariates."""
    try:
        ty = treatment_year_index(panel)
    except ValueError as exc:
        raise ValueError("Farm panel has no treated units for DVDS sensitivity") from exc

    snap = farm_level_snapshot(panel, treatment_year=ty)
    snap = snap.copy()
    snap["yield_delta"] = snap["yield_post_intervention"] - snap["yield_pre_intervention"]
    covs = [c for c in PSM_COVARIATE_COLS if c in snap.columns]
    if not covs:
        raise ValueError("No PSM covariates available on farm snapshot for DVDS")
    cols = ["received_intervention", "yield_delta", *covs]
    work = snap[cols].dropna()
    if work.empty:
        raise ValueError("No complete farm rows for DVDS after dropping missing covariates")
    if int((work["received_intervention"] == 1).sum()) == 0:
        raise ValueError("DVDS requires at least one treated farm")
    if int((work["received_intervention"] == 0).sum()) == 0:
        raise ValueError("DVDS requires at least one control farm")
    return work


def compute_sensitivity_bounds(settings: APISettings) -> list[SensitivityBounds]:
    """
    DVDS sharp ATE bounds at each Λ in ``settings.dvds_lambda_grid`` (analytic Wald CIs).

    ``tipping_point_lambda`` is computed once and repeated on each grid element.
    """
    from api.telemetry import trace_span

    with trace_span("dvds.sensitivity"):
        return _compute_sensitivity_bounds_impl(settings)


def _compute_sensitivity_bounds_impl(settings: APISettings) -> list[SensitivityBounds]:
    panel = load_dvds_panel(settings)
    snapshot = build_dvds_snapshot(panel)
    covs = [c for c in PSM_COVARIATE_COLS if c in snapshot.columns]

    tp = tipping_point(
        snapshot,
        treatment_col="received_intervention",
        outcome_col="yield_delta",
        covariate_cols=covs,
        n_folds=5,
        random_state=42,
    )

    bounds: list[SensitivityBounds] = []
    for lam in settings.dvds_lambda_grid:
        if lam < 1.0:
            continue
        result = dvds_ate(
            snapshot,
            treatment_col="received_intervention",
            outcome_col="yield_delta",
            covariate_cols=covs,
            lambda_=lam,
            n_folds=5,
            random_state=42,
        )
        bounds.append(
            SensitivityBounds(
                lambda_=lam,
                ate_lower=result.ate_lower,
                ate_upper=result.ate_upper,
                ci_lower=result.ate_ci_lower,
                ci_upper=result.ate_ci_upper,
                tipping_point_lambda=tp,
            )
        )
    return bounds
