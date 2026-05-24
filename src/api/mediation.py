"""Intervention-path mediation: resolve canonical mediators and run g-computation."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from analysis.mediation import (
    build_intervention_mediation_frame,
    mediation_analysis,
    multi_mediator_decomposition,
)
from api.schemas import (
    MediationDecomposition,
    MediatorEffect,
    MediatorId,
    SimulateInterventionRequest,
)

if TYPE_CHECKING:
    from torch import Tensor

MEDIATOR_COLUMN: dict[MediatorId, str] = {
    "microclimate": "microclimate_index",
    "soil_moisture": "soil_moisture_delta",
    "cssvd_prevalence": "cssvd_prevalence_delta",
}

_COLUMN_MEDIATOR: dict[str, MediatorId] = {v: k for k, v in MEDIATOR_COLUMN.items()}
_DISCOVERED_DAG_PATH = Path("reports/causal/discovered_dag_latest.json")


def _annual_mean_delta(ds_factual: xr.Dataset, ds_cf: xr.Dataset, var: str) -> float:
    if var not in ds_factual or var not in ds_cf:
        return 0.0
    f = float(ds_factual[var].mean().values)
    c = float(ds_cf[var].mean().values)
    return f - c


def microclimate_index(ds_factual: xr.Dataset, ds_cf: xr.Dataset) -> float:
    """Composite annual mean Δtmean, Δvpd, Δrh_mean (standardized sum)."""
    parts: list[float] = []
    for var in ("tmean", "vpd", "rh_mean"):
        if var in ds_factual and var in ds_cf:
            parts.append(_annual_mean_delta(ds_factual, ds_cf, var))
    if not parts:
        return 0.0
    arr = np.array(parts, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
    return float(np.sum(arr) / max(std, 1e-6))


def resolve_mediator_scalars(
    ds_cf: xr.Dataset,
    ds_factual: xr.Dataset,
    *,
    biotic_baseline: dict[str, Any] | None,
    biotic_projected: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Path-level mediator values for counterfactual (0) vs factual (1) arms."""
    sm_cf = float(ds_cf["sm_root"].mean().values) if "sm_root" in ds_cf else 0.0
    sm_fact = float(ds_factual["sm_root"].mean().values) if "sm_root" in ds_factual else 0.0
    micro_cf = 0.0
    micro_fact = microclimate_index(ds_factual, ds_cf)

    cssvd_cf = 0.0
    cssvd_fact = 0.0
    if biotic_baseline is not None and biotic_projected is not None:
        la_b = biotic_baseline.get("loss_attribution") or {}
        la_p = biotic_projected.get("loss_attribution") or {}
        if hasattr(la_b, "cssvd"):
            cssvd_cf = float(la_b.cssvd)
            cssvd_fact = float(la_p.cssvd)
        else:
            cssvd_cf = float(la_b.get("cssvd", 0.0))
            cssvd_fact = float(la_p.get("cssvd", 0.0))

    cf_vals = {
        "microclimate_index": micro_cf,
        "soil_moisture_delta": 0.0,
        "cssvd_prevalence_delta": 0.0,
    }
    fact_vals = {
        "microclimate_index": micro_fact,
        "soil_moisture_delta": sm_fact - sm_cf,
        "cssvd_prevalence_delta": cssvd_fact - cssvd_cf,
    }
    return cf_vals, fact_vals


def _use_discovered_dag() -> bool:
    return os.getenv("MEDIATION_USE_DISCOVERED_DAG", "").lower() in {"1", "true", "yes", "on"}


def _ordered_mediators_from_discovered_dag(
    requested: Sequence[MediatorId],
    path: Path = _DISCOVERED_DAG_PATH,
) -> list[MediatorId] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_edges = payload.get("edges", payload)
    requested_set = set(requested)
    ordered: list[MediatorId] = []
    for edge in raw_edges:
        if isinstance(edge, dict):
            dst = str(edge.get("target") or edge.get("dst") or edge.get("to"))
        else:
            dst = str(edge[1])
        med = _COLUMN_MEDIATOR.get(dst)
        if med in requested_set and med not in ordered:
            ordered.append(med)
    for med in requested:
        if med not in ordered:
            ordered.append(med)
    return ordered


def compute_intervention_mediation(
    request: SimulateInterventionRequest,
    *,
    samples_cf: Tensor,
    samples_factual: Tensor,
    ds_cf: xr.Dataset,
    ds_factual: xr.Dataset,
    biotic_baseline: dict[str, Any] | None,
    biotic_projected: dict[str, Any] | None,
    decompose_mediators: Sequence[MediatorId],
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> MediationDecomposition:
    """Run per-mediator NDE/NIE and optional multi-mediator path table."""
    from api.telemetry import trace_span

    with trace_span("mediation.decompose", intervention=request.intervention_type):
        return _compute_intervention_mediation_impl(
            request,
            samples_cf=samples_cf,
            samples_factual=samples_factual,
            ds_cf=ds_cf,
            ds_factual=ds_factual,
            biotic_baseline=biotic_baseline,
            biotic_projected=biotic_projected,
            decompose_mediators=decompose_mediators,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )


def _compute_intervention_mediation_impl(
    request: SimulateInterventionRequest,
    *,
    samples_cf: Tensor,
    samples_factual: Tensor,
    ds_cf: xr.Dataset,
    ds_factual: xr.Dataset,
    biotic_baseline: dict[str, Any] | None,
    biotic_projected: dict[str, Any] | None,
    decompose_mediators: Sequence[MediatorId],
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> MediationDecomposition:
    lat = request.farm_location.lat
    lon = request.farm_location.lon
    dag_source = "assumed"
    ordered_mediators = list(decompose_mediators)
    if _use_discovered_dag():
        discovered_order = _ordered_mediators_from_discovered_dag(decompose_mediators)
        if discovered_order is not None:
            ordered_mediators = discovered_order
            dag_source = "discovered"
    cf_vals, fact_vals = resolve_mediator_scalars(
        ds_cf,
        ds_factual,
        biotic_baseline=biotic_baseline,
        biotic_projected=biotic_projected,
    )
    covariate_row = {
        "lat": lat,
        "lon": lon,
        "farm_size_ha": request.farm_size_ha,
    }

    per_mediator: list[MediatorEffect] = []
    frames: list[pd.DataFrame] = []

    for med_id in ordered_mediators:
        col = MEDIATOR_COLUMN[med_id]
        med_cf = {col: cf_vals[col]}
        med_fact = {col: fact_vals[col]}
        frame = build_intervention_mediation_frame(
            samples_cf=samples_cf.detach().cpu().numpy().reshape(-1),
            samples_factual=samples_factual.detach().cpu().numpy().reshape(-1),
            mediator_values_cf=med_cf,
            mediator_values_factual=med_fact,
            covariate_row=covariate_row,
        )
        frame = frame.rename(columns={"yield": "outcome"})
        res = mediation_analysis(
            frame,
            treatment_col="treatment",
            outcome_col="outcome",
            mediator_col=col,
            covariate_cols=list(covariate_row.keys()),
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        per_mediator.append(
            MediatorEffect(
                mediator=med_id,
                nde=res.nde,
                nie=res.nie,
                total_effect=res.total_effect,
                proportion_mediated=res.proportion_mediated,
                nde_ci=res.nde_ci,
                nie_ci=res.nie_ci,
                rho_critical=res.rho_critical,
            )
        )
        full_frame = build_intervention_mediation_frame(
            samples_cf=samples_cf.detach().cpu().numpy().reshape(-1),
            samples_factual=samples_factual.detach().cpu().numpy().reshape(-1),
            mediator_values_cf=cf_vals,
            mediator_values_factual=fact_vals,
            covariate_row=covariate_row,
        )
        full_frame = full_frame.rename(columns={"yield": "outcome"})
        frames.append(full_frame)

    path_table: list[dict[str, Any]] = []
    if len(ordered_mediators) > 1:
        cols = [MEDIATOR_COLUMN[m] for m in ordered_mediators]
        combined = frames[-1] if frames else pd.DataFrame()
        if not combined.empty and all(c in combined.columns for c in cols):
            table = multi_mediator_decomposition(
                combined,
                treatment_col="treatment",
                outcome_col="outcome",
                mediator_cols=cols,
                covariate_cols=list(covariate_row.keys()),
                n_bootstrap=max(50, n_bootstrap // 2),
                random_state=random_state,
            )
            path_table = table.to_dict(orient="records")

    from api import metrics as prom_metrics

    if per_mediator:
        first = per_mediator[0]
        if first.nde and abs(first.nde) > 1e-9:
            prom_metrics.set_mediation_ratio(
                request.intervention_type,
                float(first.nie / first.nde) if first.nie is not None else 0.0,
            )

    return MediationDecomposition(
        per_mediator=per_mediator,
        dag_source=dag_source,
        path_table=path_table,
    )


__all__ = [
    "MEDIATOR_COLUMN",
    "compute_intervention_mediation",
    "microclimate_index",
    "resolve_mediator_scalars",
]
