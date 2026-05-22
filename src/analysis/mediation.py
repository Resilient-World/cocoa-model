"""
Causal mediation analysis (Imai, Keele & Yamamoto 2010; VanderWeele 2015).

Natural direct effects (NDE) and natural indirect effects (NIE) via cross-fitted
g-computation with HistGradientBoosting nuisances (consistent with heterogeneity.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from analysis.heterogeneity import default_nuisance_models

PS_CLIP = (0.01, 0.99)
DEFAULT_RHO_GRID = 19


@dataclass(frozen=True)
class MediationResult:
    """Single-mediator decomposition with bootstrap CIs."""

    nde: float
    nie: float
    total_effect: float
    proportion_mediated: float
    nde_ci: tuple[float, float]
    nie_ci: tuple[float, float]
    total_effect_ci: tuple[float, float]
    rho_critical: float | None = None
    sensitivity_curve: list[dict[str, float]] = field(default_factory=list)


def _check_binary_treatment(series: pd.Series) -> None:
    vals = set(pd.unique(series.dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError("treatment_col must be binary 0/1")


def _crossfit_mediation_nuisances(
    x: np.ndarray,
    t: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted predictions for e(M|X,T), e(Y|X,T,M), e(T|X)."""
    n = len(y)
    m_hat = np.zeros(n, dtype=float)
    y_hat = np.zeros(n, dtype=float)
    t_prob = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    reg, clf = default_nuisance_models(random_state)

    for tr, te in kf.split(x):
        g_m0 = clone(reg)
        g_m1 = clone(reg)
        g_y = clone(reg)
        g_t = clone(clf)

        g_t.fit(x[tr], t[tr])
        t_prob[te] = g_t.predict_proba(x[te])[:, 1]

        g_m0.fit(x[tr], m[tr], sample_weight=1.0 - t[tr])
        g_m1.fit(x[tr], m[tr], sample_weight=t[tr])
        m_hat[te] = (1.0 - t[te]) * g_m0.predict(x[te]) + t[te] * g_m1.predict(x[te])

        g_y.fit(
            np.column_stack([x[tr], t[tr], m[tr]]),
            y[tr],
        )
        y_hat[te] = g_y.predict(np.column_stack([x[te], t[te], m_hat[te]]))

    t_prob = np.clip(t_prob, PS_CLIP[0], PS_CLIP[1])
    return m_hat, y_hat, t_prob


def _gcomp_effects_single_mediator(
    x: np.ndarray,
    t: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit NDE, NIE, and total effect contributions (vectorized predictions)."""
    n = len(y)
    m0 = np.zeros(n, dtype=float)
    m1 = np.zeros(n, dtype=float)
    y00 = np.zeros(n, dtype=float)
    y10 = np.zeros(n, dtype=float)
    y11 = np.zeros(n, dtype=float)
    t0 = np.zeros((n, 1), dtype=float)
    t1 = np.ones((n, 1), dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    reg, clf = default_nuisance_models(random_state)
    n_cov = x.shape[1]

    for tr, te in kf.split(x):
        g_m = clone(reg)
        g_y = clone(reg)
        g_t = clone(clf)

        x_tr, x_te = x[tr], x[te]
        if n_cov > 0:
            g_t.fit(x_tr, t[tr])
            xm_tr = np.column_stack([x_tr, t[tr].reshape(-1, 1)])
            xm_te_0 = np.column_stack([x_te, np.zeros((len(te), 1))])
            xm_te_1 = np.column_stack([x_te, np.ones((len(te), 1))])
            xy_tr = np.column_stack([x_tr, t[tr], m[tr]])
            xy_te_0 = lambda mvec: np.column_stack(
                [x_te, np.zeros((len(te), 1)), mvec.reshape(-1, 1)]
            )
            xy_te_1 = lambda mvec: np.column_stack(
                [x_te, np.ones((len(te), 1)), mvec.reshape(-1, 1)]
            )
        else:
            xm_tr = t[tr].reshape(-1, 1)
            xm_te_0 = np.zeros((len(te), 1))
            xm_te_1 = np.ones((len(te), 1))
            xy_tr = np.column_stack([t[tr], m[tr]])
            xy_te_0 = lambda mvec: np.column_stack([np.zeros((len(te), 1)), mvec.reshape(-1, 1)])
            xy_te_1 = lambda mvec: np.column_stack([np.ones((len(te), 1)), mvec.reshape(-1, 1)])

        g_m.fit(xm_tr, m[tr])
        g_y.fit(xy_tr, y[tr])
        m0[te] = g_m.predict(xm_te_0)
        m1[te] = g_m.predict(xm_te_1)
        y00[te] = g_y.predict(xy_te_0(m0[te]))
        y10[te] = g_y.predict(xy_te_1(m0[te]))
        y11[te] = g_y.predict(xy_te_1(m1[te]))

    nde_i = y10 - y00
    nie_i = y11 - y10
    te_i = y11 - y00
    return nde_i, nie_i, te_i


def _rho_sensitivity_nie(
    nie: float,
    m: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    *,
    n_grid: int = DEFAULT_RHO_GRID,
) -> tuple[float | None, list[dict[str, float]]]:
    """
    Sensitivity of NIE to mediator-outcome confounding (VanderWeele 2015 spirit).

    Adjusted NIE(ρ) = NIE - ρ * sd(M) * sd(residual_Y).
    """
    resid = y - HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(
        np.column_stack([x, t, m]), y
    ).predict(np.column_stack([x, t, m]))
    scale = float(np.std(m, ddof=1) * np.std(resid, ddof=1))
    curve: list[dict[str, float]] = []
    rho_critical: float | None = None
    for rho in np.linspace(0.0, 0.9, n_grid):
        adj = float(nie - rho * scale)
        curve.append({"rho": float(rho), "nie_adjusted": adj})
        if rho_critical is None and adj <= 0.0:
            rho_critical = float(rho)
    return rho_critical, curve


def mediation_analysis(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    mediator_col: str,
    covariate_cols: Sequence[str],
    n_bootstrap: int = 500,
    random_state: int = 42,
    n_folds: int = 5,
) -> MediationResult:
    """
    Estimate natural direct and indirect effects via cross-fitted g-computation.

    Parameters
    ----------
    df
        Panel with binary treatment, outcome, mediator, and covariates.
    treatment_col, outcome_col, mediator_col
        Column names for ``T``, ``Y``, and ``M``.
    covariate_cols
        Confounders ``X`` (may be empty).
    n_bootstrap
        Percentile bootstrap replications for CIs.
    random_state
        RNG seed for folds and bootstrap.
    n_folds
        Cross-fitting folds for nuisance models.

    Returns
    -------
    MediationResult
        Point estimates, bootstrap CIs, and rho sensitivity curve.

    Notes
    -----
    Implements the Imai, Keele and Yamamoto (2010) g-computation estimator with
    HistGradientBoosting nuisances (VanderWeele 2015 for rho sensitivity).
    """
    cols = list(covariate_cols)
    for c in (treatment_col, outcome_col, mediator_col, *cols):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")
    use_cols = [treatment_col, outcome_col, mediator_col, *cols]
    work = df[use_cols].dropna()
    if len(work) < max(30, n_folds * 5):
        raise ValueError(f"Need at least {max(30, n_folds * 5)} complete rows for mediation")

    _check_binary_treatment(work[treatment_col])
    x = work[cols].astype(float).to_numpy() if cols else np.zeros((len(work), 0), dtype=float)
    t = work[treatment_col].astype(int).to_numpy()
    m = work[mediator_col].astype(float).to_numpy()
    y = work[outcome_col].astype(float).to_numpy()

    nde_i, nie_i, te_i = _gcomp_effects_single_mediator(
        x, t, m, y, n_folds=n_folds, random_state=random_state
    )
    nde = float(np.mean(nde_i))
    nie = float(np.mean(nie_i))
    te = float(np.mean(te_i))
    prop = float(nie / te) if abs(te) > 1e-9 else 0.0

    rng = np.random.default_rng(random_state)
    n = len(work)
    boot_nde: list[float] = []
    boot_nie: list[float] = []
    boot_te: list[float] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_nde.append(float(np.mean(nde_i[idx])))
        boot_nie.append(float(np.mean(nie_i[idx])))
        boot_te.append(float(np.mean(te_i[idx])))

    nde_ci = tuple(np.percentile(boot_nde, [2.5, 97.5]).astype(float))
    nie_ci = tuple(np.percentile(boot_nie, [2.5, 97.5]).astype(float))
    te_ci = tuple(np.percentile(boot_te, [2.5, 97.5]).astype(float))

    rho_crit, curve = _rho_sensitivity_nie(nie, m, y, t, x)

    return MediationResult(
        nde=nde,
        nie=nie,
        total_effect=te,
        proportion_mediated=prop,
        nde_ci=(float(nde_ci[0]), float(nde_ci[1])),
        nie_ci=(float(nie_ci[0]), float(nie_ci[1])),
        total_effect_ci=(float(te_ci[0]), float(te_ci[1])),
        rho_critical=rho_crit,
        sensitivity_curve=curve,
    )


def multi_mediator_decomposition(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    mediator_cols: Sequence[str],
    covariate_cols: Sequence[str],
    n_bootstrap: int = 200,
    random_state: int = 42,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Path-specific effects for ordered mediators (e.g. microclimate → soil → CSSVD).

    Returns a table with columns: path, effect, ci_low, ci_high, share_of_total.
    """
    mediators = list(mediator_cols)
    if not mediators:
        raise ValueError("mediator_cols must be non-empty")

    single = mediation_analysis(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        mediator_col=mediators[0],
        covariate_cols=covariate_cols,
        n_bootstrap=max(50, n_bootstrap // 2),
        random_state=random_state,
        n_folds=n_folds,
    )
    total = single.total_effect if abs(single.total_effect) > 1e-9 else 1.0

    rows: list[dict[str, Any]] = []
    for med in mediators:
        res = mediation_analysis(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            mediator_col=med,
            covariate_cols=covariate_cols,
            n_bootstrap=max(50, n_bootstrap // max(1, len(mediators))),
            random_state=random_state + hash(med) % 1000,
            n_folds=n_folds,
        )
        rows.append(
            {
                "path": f"T->{med}->Y",
                "effect": res.nie,
                "ci_low": res.nie_ci[0],
                "ci_high": res.nie_ci[1],
                "share_of_total": float(res.nie / total),
            }
        )
        rows.append(
            {
                "path": f"T->{med} (direct)",
                "effect": res.nde,
                "ci_low": res.nde_ci[0],
                "ci_high": res.nde_ci[1],
                "share_of_total": float(res.nde / total),
            }
        )

    rows.append(
        {
            "path": "total",
            "effect": single.total_effect,
            "ci_low": single.total_effect_ci[0],
            "ci_high": single.total_effect_ci[1],
            "share_of_total": 1.0,
        }
    )
    return pd.DataFrame(rows)


def build_intervention_mediation_frame(
    *,
    samples_cf: np.ndarray,
    samples_factual: np.ndarray,
    mediator_values_cf: dict[str, float],
    mediator_values_factual: dict[str, float],
    covariate_row: dict[str, float],
) -> pd.DataFrame:
    """
    Build a pseudo-panel for single-farm mediation from MC yield draws.

    Each MC index contributes two rows (control vs treated paths) with shared
    mediator scalars and covariates.
    """
    n = min(len(samples_cf), len(samples_factual))
    rows: list[dict[str, float]] = []
    for i in range(n):
        base = dict(covariate_row)
        for t_val, y_val, med_dict in (
            (0, float(samples_cf[i]), mediator_values_cf),
            (1, float(samples_factual[i]), mediator_values_factual),
        ):
            row = {"treatment": float(t_val), "yield": y_val, **base}
            row.update(med_dict)
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "MediationResult",
    "build_intervention_mediation_frame",
    "mediation_analysis",
    "multi_mediator_decomposition",
]
