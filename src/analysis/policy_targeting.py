"""
Policy targeting utilities for cooperative-level rollout planning.

These functions are matplotlib-free: they return DataFrames that can be plotted
by the caller (frontend/notebooks).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from analysis.heterogeneity import CATEResult

PS_CLIP = (0.01, 0.99)


def rank_farms_by_uplift(
    cate_result: CATEResult,
    *,
    intervention_cost_usd_per_farm: float,
    cocoa_price_usd: float,
    farm_areas_ha: pd.Series,
) -> pd.DataFrame:
    """
    Rank farms by expected net uplift USD.

    Net uplift (USD) = max(tau_hat, 0) * area_ha * cocoa_price_usd - intervention_cost
    Ties are broken by lower standard error (more certain uplift first).
    """
    tau = cate_result.tau_hat
    se = cate_result.se.reindex(tau.index)
    area = farm_areas_ha.reindex(tau.index).astype(float)

    avoided_tonnes = np.maximum(tau.to_numpy(dtype=float), 0.0) * area.to_numpy(dtype=float)
    gross_usd = avoided_tonnes * float(cocoa_price_usd)
    net_usd = gross_usd - float(intervention_cost_usd_per_farm)

    out = pd.DataFrame(
        {
            "tau_hat_tonnes_per_ha": tau.astype(float),
            "se": se.astype(float),
            "area_ha": area,
            "avoided_loss_tonnes": avoided_tonnes,
            "gross_uplift_usd": gross_usd,
            "net_uplift_usd": net_usd,
        },
        index=tau.index,
    )
    return out.sort_values(by=["net_uplift_usd", "se"], ascending=[False, True])


def policy_value_curve(
    ranked: pd.DataFrame,
    *,
    uplift_col: str = "avoided_loss_tonnes",
) -> pd.DataFrame:
    """
    Cumulative value curve for targeting the top-K farms.

    Parameters
    ----------
    ranked:
        Output from :func:`rank_farms_by_uplift` (already sorted).
    uplift_col:
        Column to cumulate, e.g. ``avoided_loss_tonnes`` or ``net_uplift_usd``.
    """
    if uplift_col not in ranked.columns:
        raise ValueError(f"Missing uplift column '{uplift_col}'")
    vals = ranked[uplift_col].to_numpy(dtype=float)
    cum = np.cumsum(np.maximum(vals, 0.0))
    return pd.DataFrame(
        {
            "k": np.arange(1, len(ranked) + 1),
            "cumulative_value": cum,
        }
    )


def optimal_targeting_policy(
    cate_estimates: pd.Series | np.ndarray,
    costs_per_farm: pd.Series | np.ndarray,
    budget: float,
) -> np.ndarray:
    """
    Greedy budgeted targeting by CATE / cost ratio.

    Returns a boolean mask (same length as inputs) indicating selected units.
    """
    tau = np.asarray(cate_estimates, dtype=float).ravel()
    costs = np.asarray(costs_per_farm, dtype=float).ravel()
    if len(tau) != len(costs):
        raise ValueError("cate_estimates and costs_per_farm must have the same length")
    if budget <= 0:
        return np.zeros(len(tau), dtype=bool)

    safe_costs = np.where(costs > 0, costs, np.inf)
    score = tau / safe_costs
    order = np.argsort(-score)
    selected = np.zeros(len(tau), dtype=bool)
    spent = 0.0
    for i in order:
        c = costs[i]
        if not np.isfinite(c) or c <= 0:
            continue
        if spent + c <= budget:
            selected[i] = True
            spent += c
    return selected


def _crossfit_nuisances_policy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted outcome and propensity nuisances for DR policy evaluation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    m_hat = np.zeros_like(y, dtype=float)
    e_hat = np.zeros_like(t, dtype=float)
    reg = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=6, random_state=random_state
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6, random_state=random_state
    )
    for tr, te in kf.split(x):
        m = clone(reg)
        g = clone(clf)
        m.fit(x[tr], y[tr])
        m_hat[te] = m.predict(x[te])
        g.fit(x[tr], t[tr])
        e_hat[te] = g.predict_proba(x[te])[:, 1]
    e_hat = np.clip(e_hat, PS_CLIP[0], PS_CLIP[1])
    return m_hat, e_hat


def doubly_robust_policy_value(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    policy_mask: np.ndarray | pd.Series,
    tau_hat: np.ndarray | pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> float:
    """
    Doubly-robust estimate of the policy value for targeting rule ``policy_mask``.

    V(pi) = E[ pi(X) * tau(X) ] + E[ (pi*T - pi*e) / (e(1-e)) * (Y - m - tau*T) ]
    """
    pi = np.asarray(policy_mask, dtype=float).ravel()
    tau = np.asarray(tau_hat, dtype=float).ravel()
    y = df[outcome_col].astype(float).to_numpy()
    t = df[treatment_col].astype(int).to_numpy()
    x = df[list(covariate_cols)].astype(float).to_numpy()
    if len(pi) != len(y) or len(tau) != len(y):
        raise ValueError("policy_mask and tau_hat must align with df rows")

    m_hat, e_hat = _crossfit_nuisances_policy(
        x, y, t, n_folds=n_folds, random_state=random_state
    )
    direct = pi * tau
    resid = y - m_hat - tau * t
    ipw_correction = (pi * t - pi * e_hat) / (e_hat * (1.0 - e_hat)) * resid
    return float(np.mean(direct + ipw_correction))


def targeting_from_cate(
    cate_result: CATEResult,
    costs_per_farm: pd.Series,
    budget: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build greedy targeting mask and ranked table from :class:`CATEResult`.

    Returns ``(policy_mask, ranked_df)`` where ``ranked_df`` includes
    ``selected`` and ``cate_per_cost`` columns.
    """
    tau = cate_result.tau_hat
    costs = costs_per_farm.reindex(tau.index).astype(float)
    mask = optimal_targeting_policy(tau, costs, budget)
    safe_costs = costs.replace(0, np.nan)
    ranked = pd.DataFrame(
        {
            "tau_hat": tau,
            "cost": costs,
            "cate_per_cost": tau / safe_costs,
            "selected": mask,
        },
        index=tau.index,
    )
    ranked = ranked.sort_values(
        ["selected", "cate_per_cost"],
        ascending=[False, False],
        na_position="last",
    )
    return mask, ranked


__all__ = [
    "rank_farms_by_uplift",
    "policy_value_curve",
    "optimal_targeting_policy",
    "doubly_robust_policy_value",
    "targeting_from_cate",
]
