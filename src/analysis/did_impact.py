"""
Difference-in-Differences (DiD) impact and financial valuation for matched farm panels.

Works with output from :func:`analysis.psm_matching.propensity_score_match`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import pandas as pd

MatchRole = Literal["treated", "control"]


class DiDResult(NamedTuple):
    """Average treatment effect on the treated (ATT) from a matched DiD design."""

    att: float
    treated_change_mean: float
    control_change_mean: float
    n_pairs: int


class AvoidedRevenueResult(NamedTuple):
    """Financial value of avoided yield loss for the treated cohort."""

    total_avoided_revenue_usd: float
    att_tonnes_per_ha: float
    cocoa_price_usd: float
    n_treated_farms: int
    per_farm_revenue_usd: pd.Series


def _resolve_role_mask(
    df: pd.DataFrame,
    *,
    role_col: str,
    treatment_col: str,
) -> pd.Series:
    if role_col in df.columns:
        return df[role_col] == "treated"
    if treatment_col in df.columns:
        return df[treatment_col] == 1
    raise ValueError(f"Need '{role_col}' or '{treatment_col}' column to identify treated farms")


def calculate_did_att(
    matched_df: pd.DataFrame,
    *,
    yield_pre_col: str = "yield_pre_intervention",
    yield_post_col: str = "yield_post_intervention",
    match_pair_id_col: str = "match_pair_id",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
) -> DiDResult:
    """
    Estimate the Average Treatment effect on the Treated (ATT) using DiD on matched pairs.

    For each matched pair ``i``:

    .. math::

        \\widehat{\\tau}_i = (Y^{T}_{i,post} - Y^{T}_{i,pre})
                           - (Y^{C}_{i,post} - Y^{C}_{i,pre})

    The ATT is the mean of :math:`\\widehat{\\tau}_i` across pairs (tonnes/ha if yields
    are in tonnes per hectare).

    Parameters
    ----------
    matched_df:
        PSM-matched panel with treated and control rows per ``match_pair_id``.
    yield_pre_col, yield_post_col:
        Outcome columns before and after the intervention.
    match_pair_id_col:
        Identifier linking treated-control pairs.
    role_col:
        Column with values ``\"treated\"`` / ``\"control\"`` (from PSM output).
    treatment_col:
        Fallback binary treatment column if ``role_col`` is absent.

    Returns
    -------
    DiDResult
        ``att`` (ATT), mean outcome changes for treated/control, and pair count.
    """
    required = {yield_pre_col, yield_post_col, match_pair_id_col}
    missing = required - set(matched_df.columns)
    if missing:
        raise ValueError(f"Matched DataFrame missing columns: {sorted(missing)}")

    if matched_df[[yield_pre_col, yield_post_col]].isna().any().any():
        raise ValueError("Yield columns contain missing values")

    work = matched_df.copy()
    work["_delta"] = work[yield_post_col] - work[yield_pre_col]

    if role_col in work.columns:
        treated_mask = work[role_col] == "treated"
        control_mask = work[role_col] == "control"
    else:
        treated_mask = _resolve_role_mask(work, role_col=role_col, treatment_col=treatment_col)
        control_mask = ~treated_mask

    treated_delta = work.loc[treated_mask, [match_pair_id_col, "_delta"]].rename(
        columns={"_delta": "treated_delta"}
    )
    control_delta = work.loc[control_mask, [match_pair_id_col, "_delta"]].rename(
        columns={"_delta": "control_delta"}
    )

    pairs = treated_delta.merge(control_delta, on=match_pair_id_col, how="inner")
    if pairs.empty:
        raise ValueError("No complete treated-control pairs found for DiD estimation")

    if len(pairs) != treated_delta.shape[0] or len(pairs) != control_delta.shape[0]:
        raise ValueError(
            "Each match_pair_id must have exactly one treated and one control row"
        )

    pair_effects = pairs["treated_delta"] - pairs["control_delta"]
    treated_change_mean = float(pairs["treated_delta"].mean())
    control_change_mean = float(pairs["control_delta"].mean())
    att = float(pair_effects.mean())

    return DiDResult(
        att=att,
        treated_change_mean=treated_change_mean,
        control_change_mean=control_change_mean,
        n_pairs=len(pairs),
    )


def calculate_avoided_revenue_loss(
    att: float,
    matched_df: pd.DataFrame,
    cocoa_price_usd: float,
    *,
    farm_size_col: str = "farm_size_ha",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
) -> AvoidedRevenueResult:
    """
    Convert ATT (avoided yield loss, tonnes/ha) into total avoided revenue for treated farms.

    For each treated farm ``j``:

    .. math::

        \\text{avoided\\_revenue}_j = \\mathrm{ATT} \\times \\mathrm{farm\\_size\\_ha}_j
                                     \\times \\mathrm{cocoa\\_price\\_usd}

    Parameters
    ----------
    att:
        Average treatment on the treated (tonnes/ha), from :func:`calculate_did_att`.
    matched_df:
        Matched panel containing treated farms and ``farm_size_col``.
    cocoa_price_usd:
        Market cocoa price in USD per tonne.
    farm_size_col:
        Farm area in hectares.
    role_col, treatment_col:
        Used to select the treated cohort.

    Returns
    -------
    AvoidedRevenueResult
        Total USD and per-farm breakdown for the treated cohort.
    """
    if cocoa_price_usd < 0:
        raise ValueError(f"cocoa_price_usd must be non-negative, got {cocoa_price_usd}")
    if farm_size_col not in matched_df.columns:
        raise ValueError(f"Column '{farm_size_col}' not found in matched DataFrame")

    treated = matched_df[_resolve_role_mask(matched_df, role_col=role_col, treatment_col=treatment_col)].copy()
    if treated.empty:
        raise ValueError("No treated farms found in matched DataFrame")

    if treated[farm_size_col].isna().any():
        raise ValueError(f"Missing values in '{farm_size_col}' for treated farms")

    per_farm = att * treated[farm_size_col] * cocoa_price_usd
    per_farm.index = treated.index
    per_farm.name = "avoided_revenue_usd"

    return AvoidedRevenueResult(
        total_avoided_revenue_usd=float(per_farm.sum()),
        att_tonnes_per_ha=float(att),
        cocoa_price_usd=float(cocoa_price_usd),
        n_treated_farms=len(treated),
        per_farm_revenue_usd=per_farm,
    )
