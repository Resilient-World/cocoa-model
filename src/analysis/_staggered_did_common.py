"""
Shared utilities for staggered difference-in-differences (Callaway-Sant'Anna, BJS).

Panel schema, doubly-robust nuisances (HistGradientBoosting), multiplier bootstrap,
and TWFE benchmark for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

PS_CLIP = (0.01, 0.99)
Z_975 = 1.96


@dataclass
class StaggeredPanel:
    """Validated long panel with cohort and treatment indicators."""

    df: pd.DataFrame
    unit_col: str
    time_col: str
    treat_time_col: str
    outcome_col: str
    covariate_cols: list[str]
    cohort: np.ndarray  # G_i, nan = never treated
    times: np.ndarray
    cohorts: np.ndarray  # distinct finite treatment times
    unit_ids: np.ndarray


def prepare_staggered_panel(
    df: pd.DataFrame,
    *,
    unit_col: str = "farm_id",
    time_col: str = "period",
    treat_time_col: str = "treatment_period",
    outcome_col: str = "yield",
    covariate_cols: Sequence[str] | None = None,
) -> StaggeredPanel:
    """Validate panel and attach cohort ``G_i`` (first treatment time; NaN = never treated)."""
    required = {unit_col, time_col, treat_time_col, outcome_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")

    work = df.copy()
    if work[outcome_col].isna().any():
        raise ValueError(f"Outcome '{outcome_col}' contains missing values")

    if covariate_cols is None:
        exclude = {
            unit_col,
            time_col,
            treat_time_col,
            outcome_col,
            "received_intervention",
            "treated",
            "match_role",
        }
        covariate_cols = [
            c for c in work.columns if c not in exclude and pd.api.types.is_numeric_dtype(work[c])
        ]
    covs = list(covariate_cols)

    g_series = work.groupby(unit_col)[treat_time_col].min().rename("_G")
    work = work.merge(g_series, on=unit_col, how="left")
    cohort = work["_G"].to_numpy(dtype=float)
    never = work.groupby(unit_col)[treat_time_col].apply(lambda s: s.isna().all())
    never_ids = set(never[never].index)
    work.loc[work[unit_col].isin(never_ids), "_G"] = np.nan

    times = np.sort(work[time_col].unique())
    finite_g = work.loc[work["_G"].notna(), "_G"].unique()
    cohorts = np.sort(finite_g.astype(float))

    return StaggeredPanel(
        df=work,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
        outcome_col=outcome_col,
        covariate_cols=covs,
        cohort=cohort,
        times=times,
        cohorts=cohorts,
        unit_ids=work[unit_col].unique(),
    )


def is_staggered(panel: pd.DataFrame, treat_time_col: str, unit_col: str) -> bool:
    """True when >1 distinct first-treatment time among eventually treated units."""
    g = panel.groupby(unit_col)[treat_time_col].min()
    treated = g.dropna()
    if treated.empty:
        return False
    return treated.nunique() > 1


def control_mask_at_t(
    panel: StaggeredPanel,
    t: float | int,
) -> np.ndarray:
    """Never-treated or not-yet-treated at calendar time ``t`` (CS control group)."""
    g = panel.df["_G"].to_numpy(dtype=float)
    time = panel.df[panel.time_col].to_numpy()
    never = np.isnan(g)
    not_yet = g > float(t)
    return (time == float(t)) & (never | not_yet)


def treated_cohort_mask_at_t(
    panel: StaggeredPanel,
    g: float | int,
    t: float | int,
) -> np.ndarray:
    """Units in cohort ``g`` observed at calendar time ``t``."""
    if float(t) < float(g):
        return np.zeros(len(panel.df), dtype=bool)
    gi = panel.df["_G"].to_numpy(dtype=float)
    time = panel.df[panel.time_col].to_numpy()
    return (time == float(t)) & (gi == float(g))


def fit_dr_nuisances(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    *,
    ps_clip: tuple[float, float] = PS_CLIP,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit propensity ``e(X)`` and outcome models ``m_0(X)``, ``m_1(X)``.

    ``D=1`` denotes treated (cohort) units; ``D=0`` controls.
    """
    if len(Y) < 4 or D.sum() < 1 or (1 - D).sum() < 1:
        n = len(Y)
        return np.full(n, 0.5), Y.copy(), Y.copy()

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=random_state,
    )
    clf.fit(X, D.astype(int))
    e = np.clip(clf.predict_proba(X)[:, 1], ps_clip[0], ps_clip[1])

    m0 = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, random_state=random_state
    ).fit(X[D == 0], Y[D == 0])
    m1 = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, random_state=random_state
    ).fit(X[D == 1], Y[D == 1])

    mu0 = m0.predict(X)
    mu1 = m1.predict(X)
    return e, mu0, mu1


def dr_att_influence(
    Y: np.ndarray,
    D: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Doubly-robust ATT influence functions (Sant'Anna & Zhao 2020 style).

    Returns ``(att, psi)`` where ``psi`` is the influence contribution per row
    (scaled so mean(psi_treated) = att).
    """
    n = len(Y)
    p_t = float(D.mean())
    if p_t <= 0:
        return float("nan"), np.full(n, np.nan)

    term = (D * (Y - mu0) - (1 - D) * (e / (1 - e)) * (Y - mu0)) / p_t
    att = float(term[D == 1].mean()) if (D == 1).any() else float("nan")
    psi = term.copy()
    if (D == 1).any():
        psi[D == 1] -= att
    return att, psi


def cluster_se_from_influence(
    psi: np.ndarray,
    clusters: np.ndarray,
    *,
    treated_mask: np.ndarray | None = None,
) -> float:
    """Cluster-robust SE: sqrt of sum of squared cluster totals of influence."""
    if treated_mask is not None:
        psi = psi.copy()
        psi[~treated_mask] = 0.0
    cluster_sums: dict[Any, float] = {}
    for val, cl in zip(psi, clusters):
        if np.isnan(val):
            continue
        cluster_sums[cl] = cluster_sums.get(cl, 0.0) + float(val)
    vals = np.array(list(cluster_sums.values()))
    if len(vals) < 2:
        return float("nan")
    return float(np.sqrt(np.sum(vals**2)))


def multiplier_bootstrap_ci(
    att_point: float,
    psi_by_unit: dict[Any, float],
    *,
    n_boot: int = 999,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Multiplier bootstrap CI from unit-level influence totals (Callaway & Sant'Anna Alg. 1)."""
    units = list(psi_by_unit.keys())
    psi_vec = np.array([psi_by_unit[u] for u in units])
    if len(units) < 2:
        return float("nan"), float("nan"), float("nan")

    if n_boot < 1:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        mult = rng.standard_normal(len(units))
        boots[b] = att_point + float(np.sum(psi_vec * mult) / len(units))
    se = float(boots.std(ddof=1))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return se, lo, hi


def simultaneous_bootstrap_bands(
    att_points: np.ndarray,
    psi_matrix: np.ndarray,
    *,
    n_boot: int = 999,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simultaneous CI bands for a vector of ATT estimates.

    ``psi_matrix`` shape ``(n_units, n_att)`` — influence of each unit on each ATT.
    """
    n_att = att_points.shape[0]
    if n_att == 0:
        return att_points, att_points, att_points
    rng = np.random.default_rng(random_state)
    n_units = psi_matrix.shape[0]
    boots = np.zeros((n_boot, n_att))
    for b in range(n_boot):
        mult = rng.standard_normal(n_units)
        boots[b] = att_points + (psi_matrix.T @ mult) / n_units
    lo = np.quantile(boots, alpha / 2, axis=0)
    hi = np.quantile(boots, 1 - alpha / 2, axis=0)
    se = boots.std(ddof=1, axis=0)
    return se, lo, hi


def estimate_twfe(
    panel: StaggeredPanel,
    *,
    treat_indicator_col: str | None = None,
) -> float:
    """
    Two-way fixed effects DiD (TWFE) with ``D_it = 1{t >= G_i}`` for treated units.

    Benchmark for bias tests; requires ``linearmodels``.
    """
    from linearmodels.panel import PanelOLS

    work = panel.df.copy()
    g = work["_G"]
    t = work[panel.time_col]
    if treat_indicator_col and treat_indicator_col in work.columns:
        d_it = work[treat_indicator_col].astype(float)
    else:
        d_it = ((t >= g) & g.notna()).astype(float)
    work["_D_it"] = d_it
    work = work.set_index([panel.unit_col, panel.time_col])
    y = work[panel.outcome_col]
    if panel.covariate_cols:
        exog = pd.concat([work[["_D_it"]], work[panel.covariate_cols]], axis=1)
        mod = PanelOLS(y, exog, entity_effects=True, time_effects=True)
    else:
        mod = PanelOLS(y, work[["_D_it"]], entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return float(res.params["_D_it"])


def normal_ci(att: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    z = Z_975 if alpha == 0.05 else 1.96
    if se is None or np.isnan(se) or se <= 0:
        return float("nan"), float("nan")
    return att - z * se, att + z * se
