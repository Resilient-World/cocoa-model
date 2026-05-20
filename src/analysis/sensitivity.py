"""
Sensitivity analysis for matched observational causal estimates.

- Rosenbaum (2002) bounds on hidden bias in matched-pair sign tests
- E-value for unmeasured confounding (VanderWeele & Ding 2017)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import comb

from analysis.did_impact import _pair_effects


@dataclass
class EValueResult:
    point_e_value: float
    ci_e_value: float
    estimate: float
    ci_low: float
    outcome_sd: float


def _rosenbaum_p_upper(t_plus: int, n: int, gamma: float) -> float:
    """One-sided upper bound on p-value for T+ successes under hidden bias odds ratio ``gamma``."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    denom = (1.0 + gamma) ** n
    total = 0.0
    for k in range(t_plus, n + 1):
        total += comb(n, k, exact=True) * (gamma**k) / denom
    return float(min(1.0, total))


def rosenbaum_bounds(
    matched_df: pd.DataFrame,
    *,
    gamma_range: tuple[float, float] = (1.0, 2.5),
    n_gamma: int = 16,
    yield_pre_col: str = "yield_pre_intervention",
    yield_post_col: str = "yield_post_intervention",
    match_pair_id_col: str = "match_pair_id",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity bounds for matched-pair effects.

    Returns a table with ``gamma`` and ``p_value_upper`` (worst-case one-sided p-value
    under an unobserved confounder that increases odds of treatment by at most ``gamma``).
    """
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
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    rows = [
        {"gamma": float(g), "p_value_upper": _rosenbaum_p_upper(t_plus, n, float(g))}
        for g in gammas
    ]
    return pd.DataFrame(rows)


def e_value(
    estimate: float,
    ci_low: float,
    *,
    outcome_sd: float | None = None,
    ci_side: str = "lower",
) -> EValueResult:
    """
    E-value for unmeasured confounding (VanderWeele & Ding 2017).

    Maps the additive effect and its confidence bound to a conservative risk-ratio
    scale via ``RR = exp(effect / SD)``, then applies ``E = RR + sqrt(RR * (RR - 1))``.
    """
    if outcome_sd is None or outcome_sd <= 0:
        outcome_sd = 1.0

    def _e_from_effect(effect: float) -> float:
        if effect <= 0:
            return 1.0
        rr = float(np.exp(effect / outcome_sd))
        if rr < 1.0:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1.0)))

    bound = ci_low if ci_side == "lower" else estimate
    return EValueResult(
        point_e_value=_e_from_effect(float(estimate)),
        ci_e_value=_e_from_effect(float(bound)),
        estimate=float(estimate),
        ci_low=float(ci_low),
        outcome_sd=float(outcome_sd),
    )


__all__ = ["EValueResult", "e_value", "rosenbaum_bounds"]
