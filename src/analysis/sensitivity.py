"""
Sensitivity analysis for matched observational causal estimates.

- Rosenbaum (2002) bounds on hidden bias in matched-pair sign tests
- E-value for unmeasured confounding (VanderWeele & Ding 2017)
- Negative-control outcome falsification tests
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb

from analysis.did_impact import _pair_effects


@dataclass
class EValueResult:
    point_e_value: float
    ci_e_value: float
    estimate: float
    ci_low: float
    outcome_sd: float


@dataclass
class NegativeControlResult:
    """Falsification test: treatment should not move a negative-control outcome."""

    nco_col: str
    nco_mean_treated: float
    nco_mean_control: float
    difference: float
    t_statistic: float
    p_value: float
    falsification_pass: bool
    n_treated: int
    n_control: int
    alpha: float = 0.05


def _rosenbaum_p_upper(t_plus: int, n: int, gamma: float) -> float:
    """One-sided upper bound on p-value for T+ successes under hidden bias odds ratio ``gamma``."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if n <= 0:
        return 1.0
    denom = (1.0 + gamma) ** n
    total = 0.0
    for k in range(t_plus, n + 1):
        total += comb(n, k, exact=True) * (gamma**k) / denom
    return float(min(1.0, total))


def _pair_effects_from_outcome_col(
    matched_df: pd.DataFrame,
    outcome_col: str,
    *,
    match_pair_id_col: str,
    role_col: str,
    treatment_col: str,
) -> pd.DataFrame:
    """Pair-level treated minus control contrast on a single outcome column."""
    if outcome_col not in matched_df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found")
    if match_pair_id_col not in matched_df.columns:
        raise ValueError(f"Match id column '{match_pair_id_col}' not found")

    work = matched_df.copy()
    if role_col in work.columns:
        treated_mask = work[role_col] == "treated"
        control_mask = work[role_col] == "control"
    else:
        treated_mask = work[treatment_col] == 1
        control_mask = work[treatment_col] == 0

    t = (
        work.loc[treated_mask, [match_pair_id_col, outcome_col]]
        .rename(columns={outcome_col: "treated_outcome"})
        .set_index(match_pair_id_col)
    )
    c = (
        work.loc[control_mask, [match_pair_id_col, outcome_col]]
        .rename(columns={outcome_col: "control_outcome"})
        .set_index(match_pair_id_col)
    )
    pairs = t.join(c, how="inner")
    if pairs.empty:
        raise ValueError("No complete treated-control pairs for Rosenbaum bounds")
    pairs["pair_effect"] = pairs["treated_outcome"] - pairs["control_outcome"]
    return pairs.reset_index()


def rosenbaum_bounds(
    matched_df: pd.DataFrame,
    outcome_col: str | None = None,
    gamma_grid: Sequence[float] | None = None,
    *,
    yield_pre_col: str = "yield_pre_intervention",
    yield_post_col: str = "yield_post_intervention",
    match_pair_id_col: str = "match_pair_id",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity bounds for matched-pair effects (Rosenbaum 2002).

    Parameters
    ----------
    matched_df:
        PSM output with treated/control rows per ``match_pair_id``.
    outcome_col:
        If set, use this column for pair contrasts (level outcome). If ``None``,
        use pre/post yield columns and DiD pair effects (default).
    gamma_grid:
        Grid of hidden-bias odds ratios Γ. Default: 16 points from 1.0 to 2.5.

    Returns
    -------
    DataFrame with ``gamma`` and ``p_value_upper`` (worst-case one-sided p-value).
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(1.0, 2.5, 16)
    gammas = [float(g) for g in gamma_grid]
    if not gammas or any(g <= 0 for g in gammas):
        raise ValueError("gamma_grid must contain positive values")

    if outcome_col is not None:
        pairs = _pair_effects_from_outcome_col(
            matched_df,
            outcome_col,
            match_pair_id_col=match_pair_id_col,
            role_col=role_col,
            treatment_col=treatment_col,
        )
    else:
        pairs = _pair_effects(
            matched_df,
            yield_pre_col=yield_pre_col,
            yield_post_col=yield_post_col,
            match_pair_id_col=match_pair_id_col,
            role_col=role_col,
            treatment_col=treatment_col,
            strict_pairs=True,
        )

    effects = pairs["pair_effect"].to_numpy()
    n = len(effects)
    t_plus = int((effects > 0).sum())
    return pd.DataFrame(
        [{"gamma": g, "p_value_upper": _rosenbaum_p_upper(t_plus, n, g)} for g in gammas]
    )


def rosenbaum_gamma_at_alpha(
    bounds_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> float | None:
    """Smallest Γ in ``bounds_df`` with ``p_value_upper`` > ``alpha`` (insensitive at Γ)."""
    if bounds_df.empty:
        return None
    ok = bounds_df.loc[bounds_df["p_value_upper"] > alpha]
    if ok.empty:
        return None
    return float(ok["gamma"].min())


def e_value(
    estimate: float,
    se: float,
    *,
    outcome_sd: float | None = None,
    ci_low: float | None = None,
) -> EValueResult:
    """
    E-value for unmeasured confounding (VanderWeele & Ding 2017).

    Primary call: ``e_value(ate, se)`` uses ``ci_low = ate - 1.96 * se`` for the
    confidence-limit E-value. Pass ``ci_low=`` explicitly to override.
    """
    if ci_low is None:
        ci_low = float(estimate) - 1.96 * float(se)

    if outcome_sd is None or outcome_sd <= 0:
        outcome_sd = 1.0

    def _e_from_effect(effect: float) -> float:
        if effect <= 0:
            return 1.0
        rr = float(np.exp(effect / outcome_sd))
        if rr < 1.0:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1.0)))

    return EValueResult(
        point_e_value=_e_from_effect(float(estimate)),
        ci_e_value=_e_from_effect(float(ci_low)),
        estimate=float(estimate),
        ci_low=float(ci_low),
        outcome_sd=float(outcome_sd),
    )


def negative_control_outcome_test(
    df: pd.DataFrame,
    nco_col: str,
    *,
    treatment_col: str = "received_intervention",
    alpha: float = 0.05,
) -> NegativeControlResult:
    """
    Falsification test: a negative-control outcome should not differ by treatment.

    Under valid design, ``p_value`` should exceed ``alpha`` (no spurious association).
    """
    if nco_col not in df.columns:
        raise ValueError(f"Negative-control column '{nco_col}' not found")
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found")

    work = df[[treatment_col, nco_col]].dropna()
    treated = work.loc[work[treatment_col] == 1, nco_col].to_numpy()
    control = work.loc[work[treatment_col] == 0, nco_col].to_numpy()
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("Need >=2 treated and >=2 control units for NCO test")

    t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)
    mean_t = float(np.mean(treated))
    mean_c = float(np.mean(control))
    return NegativeControlResult(
        nco_col=nco_col,
        nco_mean_treated=mean_t,
        nco_mean_control=mean_c,
        difference=mean_t - mean_c,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        falsification_pass=float(p_value) > alpha,
        n_treated=len(treated),
        n_control=len(control),
        alpha=alpha,
    )


__all__ = [
    "EValueResult",
    "NegativeControlResult",
    "e_value",
    "negative_control_outcome_test",
    "rosenbaum_bounds",
    "rosenbaum_gamma_at_alpha",
]
