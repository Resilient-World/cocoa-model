"""
Parallel-trends diagnostics for farm panels.

- Pre-treatment placebo DiD (pseudo-effects k periods before adoption)
- Goodman–Bacon decomposition for staggered / heterogeneous adoption timing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.did_impact import _paired_bootstrap_ci


@dataclass
class PlaceboPretrendResult:
    """Summary of pre-treatment placebo DiD tests."""

    table: pd.DataFrame
    joint_pretrend_ok: bool
    max_abs_placebo_att: float


def _infer_treatment_split_year(
    panel_df: pd.DataFrame,
    *,
    time_col: str,
    treatment_col: str,
    treatment_year: int | None,
) -> int:
    if treatment_year is not None:
        return int(treatment_year)
    treated = panel_df.loc[panel_df[treatment_col] == 1]
    if treated.empty:
        raise ValueError("No treated observations; cannot infer treatment year")
    return int(treated[time_col].min())


def placebo_pretreatment_did(
    panel_df: pd.DataFrame,
    *,
    unit_col: str = "farm_id",
    time_col: str = "year",
    outcome_col: str = "yield_tonnes_per_ha",
    treatment_col: str = "received_intervention",
    treatment_year: int | None = None,
    k_periods: int = 3,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> PlaceboPretrendResult:
    """
    Pre-treatment placebo DiD: estimate pseudo-ATT at k periods before true adoption.

    For each ``k`` in ``1..k_periods``, uses calendar years ``(G-k-1, G-k)`` as a
    fake pre/post window, restricted to farm-years strictly before the true split ``G``.
    Under parallel trends, placebo ATTs should be near zero.
    """
    if k_periods < 1:
        raise ValueError("k_periods must be >= 1")
    required = {unit_col, time_col, outcome_col, treatment_col}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")

    split_year = _infer_treatment_split_year(
        panel_df,
        time_col=time_col,
        treatment_col=treatment_col,
        treatment_year=treatment_year,
    )
    pre_panel = panel_df.loc[panel_df[time_col] < split_year].copy()
    if pre_panel.empty:
        raise ValueError("No pre-treatment periods available for placebo tests")

    ever_treated = (
        panel_df.groupby(unit_col)[treatment_col].max().rename("ever_treated").reset_index()
    )

    rows: list[dict[str, float | int | bool | None]] = []
    for k in range(1, k_periods + 1):
        pre_y = split_year - k - 1
        post_y = split_year - k
        window = pre_panel.loc[pre_panel[time_col].isin([pre_y, post_y])]
        if window.empty:
            rows.append(
                {
                    "k": k,
                    "pre_year": pre_y,
                    "post_year": post_y,
                    "placebo_att": np.nan,
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_treated": 0,
                    "n_control": 0,
                    "placebo_ok": False,
                }
            )
            continue

        wide = window.pivot_table(
            index=unit_col,
            columns=time_col,
            values=outcome_col,
            aggfunc="mean",
        )
        if pre_y not in wide.columns or post_y not in wide.columns:
            rows.append(
                {
                    "k": k,
                    "pre_year": pre_y,
                    "post_year": post_y,
                    "placebo_att": np.nan,
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_treated": 0,
                    "n_control": 0,
                    "placebo_ok": False,
                }
            )
            continue

        wide = wide[[pre_y, post_y]].dropna()
        wide["delta"] = wide[post_y] - wide[pre_y]
        wide = wide.reset_index().merge(ever_treated, on=unit_col, how="left")

        t_delta = wide.loc[wide["ever_treated"] == 1, "delta"].to_numpy()
        c_delta = wide.loc[wide["ever_treated"] == 0, "delta"].to_numpy()
        if len(t_delta) < 2 or len(c_delta) < 2:
            rows.append(
                {
                    "k": k,
                    "pre_year": pre_y,
                    "post_year": post_y,
                    "placebo_att": np.nan,
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_treated": len(t_delta),
                    "n_control": len(c_delta),
                    "placebo_ok": False,
                }
            )
            continue

        n_t, n_c = len(t_delta), len(c_delta)
        effects_arr = np.array(
            [float(t_delta[i % n_t]) - float(c_delta[i % n_c]) for i in range(min(n_t, n_c))]
        )
        att = float(np.mean(t_delta) - np.mean(c_delta))
        se, ci_low, ci_high = _paired_bootstrap_ci(
            effects_arr,
            n_boot=n_boot,
            alpha=alpha,
            random_state=random_state + k,
        )
        placebo_ok = bool(ci_low <= 0.0 <= ci_high) if not np.isnan(ci_low) else False
        rows.append(
            {
                "k": k,
                "pre_year": pre_y,
                "post_year": post_y,
                "placebo_att": att,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_treated": n_t,
                "n_control": n_c,
                "placebo_ok": placebo_ok,
            }
        )

    table = pd.DataFrame(rows)
    valid = table["placebo_att"].notna()
    max_abs = float(table.loc[valid, "placebo_att"].abs().max()) if valid.any() else float("nan")
    joint_ok = bool(valid.all() and table.loc[valid, "placebo_ok"].all())
    return PlaceboPretrendResult(
        table=table,
        joint_pretrend_ok=joint_ok,
        max_abs_placebo_att=max_abs,
    )


def _two_group_did(
    df: pd.DataFrame,
    *,
    unit_col: str,
    treated_ids: set,
    control_ids: set,
    time_col: str,
    outcome_col: str,
    cohort_year: int,
) -> float:
    """2×2 DiD: pre = cohort_year-1, post = cohort_year."""
    pre_y, post_y = cohort_year - 1, cohort_year
    sub = df.loc[df[time_col].isin([pre_y, post_y])]
    wide = sub.pivot_table(index=unit_col, columns=time_col, values=outcome_col, aggfunc="mean")
    if pre_y not in wide.columns or post_y not in wide.columns:
        return float("nan")
    wide = wide[[pre_y, post_y]].dropna()
    wide["delta"] = wide[post_y] - wide[pre_y]

    t = wide.loc[wide.index.isin(treated_ids), "delta"]
    c = wide.loc[wide.index.isin(control_ids), "delta"]
    if len(t) < 1 or len(c) < 1:
        return float("nan")
    return float(t.mean() - c.mean())


def goodman_bacon_decomposition(
    panel_df: pd.DataFrame,
    *,
    unit_col: str = "farm_id",
    time_col: str = "year",
    outcome_col: str = "yield_tonnes_per_ha",
    treat_col: str = "received_intervention",
) -> pd.DataFrame:
    """
    Goodman–Bacon (2021) decomposition of TWFE into 2×2 DiD comparisons.

    For staggered adoption, reports each timing-group contrast, its DiD estimate,
    sample sizes, and share of the total weighted sum. With simultaneous adoption
    (one cohort), the table collapses to a single timing vs never-treated contrast.

    Reference: Goodman-Bacon (2021); Sant'Anna & Zhao (2020) for DR alternatives.
    """
    required = {unit_col, time_col, outcome_col, treat_col}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")

    df = panel_df[[unit_col, time_col, outcome_col, treat_col]].dropna().copy()
    first_treat = (
        df.loc[df[treat_col] == 1]
        .groupby(unit_col)[time_col]
        .min()
        .rename("first_treat_year")
    )
    cohorts = first_treat.reset_index()
    never = set(df[unit_col].unique()) - set(cohorts[unit_col])

    rows: list[dict[str, float | int | str]] = []
    total_weight = 0.0
    weighted_sum = 0.0

    for g_year, g_units in cohorts.groupby("first_treat_year"):
        g_year = int(g_year)
        g_ids = set(g_units[unit_col])
        n_g = len(g_ids)

        if never:
            n_c = len(never)
            weight = n_g * n_c / (n_g + n_c) ** 2
            did = _two_group_did(
                df,
                unit_col=unit_col,
                treated_ids=g_ids,
                control_ids=never,
                time_col=time_col,
                outcome_col=outcome_col,
                cohort_year=g_year,
            )
            rows.append(
                {
                    "comparison": "timing_vs_never",
                    "treated_cohort_year": g_year,
                    "control_group": "never_treated",
                    "n_treated": n_g,
                    "n_control": n_c,
                    "weight": weight,
                    "did_estimate": did,
                }
            )
            total_weight += weight
            weighted_sum += weight * did

        for h_year, h_units in cohorts.groupby("first_treat_year"):
            h_year = int(h_year)
            if h_year >= g_year:
                continue
            h_ids = set(h_units[unit_col])
            n_c = len(h_ids)
            if n_c == 0:
                continue
            weight = n_g * n_c / (n_g + n_c) ** 2
            did = _two_group_did(
                df,
                unit_col=unit_col,
                treated_ids=g_ids,
                control_ids=h_ids,
                time_col=time_col,
                outcome_col=outcome_col,
                cohort_year=g_year,
            )
            rows.append(
                {
                    "comparison": "timing_vs_already_treated",
                    "treated_cohort_year": g_year,
                    "control_group": f"cohort_{h_year}",
                    "n_treated": n_g,
                    "n_control": n_c,
                    "weight": weight,
                    "did_estimate": did,
                }
            )
            total_weight += weight
            weighted_sum += weight * did

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["weight_share"] = out["weight"] / total_weight if total_weight > 0 else 0.0
    out["twfe_reconstruction"] = weighted_sum
    return out.sort_values(["treated_cohort_year", "comparison"]).reset_index(drop=True)


__all__ = [
    "PlaceboPretrendResult",
    "goodman_bacon_decomposition",
    "placebo_pretreatment_did",
]
