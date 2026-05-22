"""
Difference-in-Differences (DiD) impact and financial valuation for matched farm panels.

Adds cluster-robust SEs, paired-bootstrap confidence intervals, an event-study
parallel-trends test, and uncertainty propagation into avoided-revenue results.

Backwards-compatible: existing call sites that read .att / .n_pairs etc. still work.
Works with output from :func:`analysis.psm_matching.propensity_score_match`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from analysis.bjs_imputation import BJSResult
    from analysis.csdid import ATTResult

import numpy as np
import pandas as pd

from analysis._staggered_did_common import is_staggered

MatchRole = Literal["treated", "control"]
DidMethod = Literal["pair_diff", "csdid", "bjs", "synthdid"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class DiDResult(NamedTuple):
    """Average treatment effect on the treated (ATT) from a matched DiD design.

    Backwards-compatible with the previous NamedTuple; new fields are appended
    so existing positional unpacks keep working if they only took the first 4.
    """

    att: float
    treated_change_mean: float
    control_change_mean: float
    n_pairs: int
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    method: str = "paired_did"


@dataclass
class EventStudyResult:
    """Pre/post leads & lags for parallel-trends inspection."""

    leads_lags: pd.DataFrame  # columns: period, coef, se, ci_low, ci_high
    pretrend_pvalue: float | None
    parallel_trends_ok: bool


class AvoidedRevenueResult(NamedTuple):
    """Financial value of avoided yield loss for the treated cohort."""

    total_avoided_revenue_usd: float
    att_tonnes_per_ha: float
    cocoa_price_usd: float
    n_treated_farms: int
    per_farm_revenue_usd: pd.Series
    total_avoided_revenue_ci_low_usd: float | None = None
    total_avoided_revenue_ci_high_usd: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _pair_effects(
    matched_df: pd.DataFrame,
    *,
    yield_pre_col: str,
    yield_post_col: str,
    match_pair_id_col: str,
    role_col: str,
    treatment_col: str,
    strict_pairs: bool,
) -> pd.DataFrame:
    """Return one row per match_pair_id with treated_delta, control_delta, pair_effect."""
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

    if strict_pairs:
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
            raise ValueError("Each match_pair_id must have exactly one treated and one control row")
    else:
        treated_delta = (
            work.loc[treated_mask]
            .groupby(match_pair_id_col)["_delta"]
            .mean()
            .rename("treated_delta")
        )
        control_delta = (
            work.loc[control_mask]
            .groupby(match_pair_id_col)["_delta"]
            .mean()
            .rename("control_delta")
        )
        pairs = pd.concat([treated_delta, control_delta], axis=1, join="inner").reset_index()
        if pairs.empty:
            raise ValueError("No complete treated-control pairs found for DiD estimation")

    pairs["pair_effect"] = pairs["treated_delta"] - pairs["control_delta"]
    return pairs


def _paired_bootstrap_ci(
    pair_effects: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    random_state: int,
) -> tuple[float, float, float]:
    """Return (se, ci_low, ci_high) from a cluster-on-pair bootstrap."""
    rng = np.random.default_rng(random_state)
    n = len(pair_effects)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = pair_effects[idx].mean(axis=1)
    se = float(boot_means.std(ddof=1))
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return se, lo, hi


def _two_sided_p_from_z(z: float) -> float:
    """Two-sided p-value for a standard-normal test statistic."""
    from math import erf, sqrt

    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))


def calculate_did_att(
    matched_df: pd.DataFrame,
    *,
    yield_pre_col: str = "yield_pre_intervention",
    yield_post_col: str = "yield_post_intervention",
    match_pair_id_col: str = "match_pair_id",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
    treat_time_col: str = "treatment_period",
    unit_col: str = "farm_id",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    strict_pairs: bool = True,
) -> DiDResult:
    """
    Estimate the Average Treatment effect on the Treated (ATT) on matched pairs,
    with a paired (pair-cluster) bootstrap SE/CI and a normal-approx p-value.

    Defaults preserve the previous public API; new uncertainty kwargs are optional.

    Emits :class:`DeprecationWarning` when staggered adoption timing is detected
    (multiple distinct ``treat_time_col`` values among treated units).
    """
    if treat_time_col in matched_df.columns and unit_col in matched_df.columns:
        if is_staggered(matched_df, treat_time_col, unit_col):
            warnings.warn(
                "Staggered treatment timing detected: pair-level DiD is not valid. "
                "Use did_estimator(..., method='csdid') or method='bjs'.",
                DeprecationWarning,
                stacklevel=2,
            )
    elif "treatment_year" in matched_df.columns and unit_col in matched_df.columns:
        if is_staggered(
            matched_df.rename(columns={"treatment_year": treat_time_col}), treat_time_col, unit_col
        ):
            warnings.warn(
                "Staggered treatment timing detected: pair-level DiD is not valid. "
                "Use did_estimator(..., method='csdid') or method='bjs'.",
                DeprecationWarning,
                stacklevel=2,
            )

    pairs = _pair_effects(
        matched_df,
        yield_pre_col=yield_pre_col,
        yield_post_col=yield_post_col,
        match_pair_id_col=match_pair_id_col,
        role_col=role_col,
        treatment_col=treatment_col,
        strict_pairs=strict_pairs,
    )

    att = float(pairs["pair_effect"].mean())
    treated_change_mean = float(pairs["treated_delta"].mean())
    control_change_mean = float(pairs["control_delta"].mean())

    se, ci_low, ci_high = _paired_bootstrap_ci(
        pairs["pair_effect"].to_numpy(),
        n_boot=n_boot,
        alpha=alpha,
        random_state=random_state,
    )
    p_value = _two_sided_p_from_z(att / se) if se and not np.isnan(se) and se > 0 else None

    return DiDResult(
        att=att,
        treated_change_mean=treated_change_mean,
        control_change_mean=control_change_mean,
        n_pairs=len(pairs),
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        method="paired_did_bootstrap",
    )


def event_study(
    panel_df: pd.DataFrame,
    *,
    unit_col: str = "farm_id",
    time_col: str = "period",
    treatment_time_col: str = "treatment_period",
    outcome_col: str = "yield",
    lead_window: int = 3,
    lag_window: int = 3,
    pretrend_alpha: float = 0.05,
) -> EventStudyResult:
    """
    Long-format event-study with unit and time fixed effects.

    Expects a tidy panel with one row per (unit, period). `treatment_period` is
    the period in which a unit was first treated (NaN for never-treated controls).
    Coefficients on event-time leads (k<0) should be statistically indistinguishable
    from zero under parallel trends.

    Requires ``statsmodels`` (``pip install statsmodels``).
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("event_study requires statsmodels. `pip install statsmodels`.") from exc

    df = panel_df.copy()
    df["event_time"] = df[time_col] - df[treatment_time_col]
    df["event_time"] = df["event_time"].where(df[treatment_time_col].notna(), other=np.nan)

    df["event_bin"] = df["event_time"].clip(lower=-lead_window, upper=lag_window)
    df["event_bin"] = df["event_bin"].astype("Int64")

    dummies = pd.get_dummies(df["event_bin"], prefix="k", dtype=float)
    if "k_-1" in dummies.columns:
        dummies = dummies.drop(columns=["k_-1"])

    unit_fe = pd.get_dummies(df[unit_col], prefix="u", drop_first=True, dtype=float)
    time_fe = pd.get_dummies(df[time_col], prefix="t", drop_first=True, dtype=float)

    X = pd.concat([dummies, unit_fe, time_fe], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome_col].astype(float)

    keep = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[keep], y.loc[keep]

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df.loc[keep, unit_col]})

    coefs = []
    for col in dummies.columns:
        if col not in model.params:
            continue
        k = int(col.replace("k_", ""))
        coef = model.params[col]
        se = model.bse[col]
        ci_low, ci_high = model.conf_int().loc[col].tolist()
        coefs.append({"period": k, "coef": coef, "se": se, "ci_low": ci_low, "ci_high": ci_high})

    leads_lags = pd.DataFrame(coefs).sort_values("period").reset_index(drop=True)

    pre_terms = [c for c in dummies.columns if int(c.replace("k_", "")) < 0 and c in model.params]
    if pre_terms:
        try:
            restriction = " = 0, ".join(pre_terms) + " = 0"
            ftest = model.f_test(restriction)
            pretrend_p = float(ftest.pvalue)
        except Exception:  # pragma: no cover
            pretrend_p = None
    else:
        pretrend_p = None

    parallel_ok = pretrend_p is None or pretrend_p > pretrend_alpha
    return EventStudyResult(
        leads_lags=leads_lags,
        pretrend_pvalue=pretrend_p,
        parallel_trends_ok=parallel_ok,
    )


def calculate_avoided_revenue_loss(
    att: float,
    matched_df: pd.DataFrame,
    cocoa_price_usd: float,
    *,
    farm_size_col: str = "farm_size_ha",
    role_col: str = "match_role",
    treatment_col: str = "received_intervention",
    att_ci: tuple[float, float] | None = None,
) -> AvoidedRevenueResult:
    """
    Convert ATT (tonnes/ha) into total avoided revenue for treated farms.

    If `att_ci` is provided (e.g. from `DiDResult.ci_low/ci_high`), the result
    also exposes a USD CI propagated linearly through farm area * price.
    """
    if cocoa_price_usd < 0:
        raise ValueError(f"cocoa_price_usd must be non-negative, got {cocoa_price_usd}")
    if farm_size_col not in matched_df.columns:
        raise ValueError(f"Column '{farm_size_col}' not found in matched DataFrame")

    treated = matched_df[
        _resolve_role_mask(matched_df, role_col=role_col, treatment_col=treatment_col)
    ].copy()
    if treated.empty:
        raise ValueError("No treated farms found in matched DataFrame")
    if treated[farm_size_col].isna().any():
        raise ValueError(f"Missing values in '{farm_size_col}' for treated farms")

    per_farm = att * treated[farm_size_col] * cocoa_price_usd
    per_farm.index = treated.index
    per_farm.name = "avoided_revenue_usd"

    total = float(per_farm.sum())
    total_low = total_high = None
    if att_ci is not None:
        lo, hi = att_ci
        total_low = float((lo * treated[farm_size_col] * cocoa_price_usd).sum())
        total_high = float((hi * treated[farm_size_col] * cocoa_price_usd).sum())

    return AvoidedRevenueResult(
        total_avoided_revenue_usd=total,
        att_tonnes_per_ha=float(att),
        cocoa_price_usd=float(cocoa_price_usd),
        n_treated_farms=len(treated),
        per_farm_revenue_usd=per_farm,
        total_avoided_revenue_ci_low_usd=total_low,
        total_avoided_revenue_ci_high_usd=total_high,
    )


def _long_to_matched_wide(
    panel_df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treat_time_col: str,
) -> pd.DataFrame:
    """Pivot two-period panel to wide pre/post for pair_diff."""
    times = sorted(panel_df[time_col].unique())
    if len(times) < 2:
        raise ValueError("pair_diff requires at least two time periods in panel")
    pre_t, post_t = times[0], times[-1]
    pre = panel_df.loc[panel_df[time_col] == pre_t, [unit_col, outcome_col, treat_time_col]].rename(
        columns={outcome_col: "yield_pre_intervention"}
    )
    post = panel_df.loc[panel_df[time_col] == post_t, [unit_col, outcome_col]].rename(
        columns={outcome_col: "yield_post_intervention"}
    )
    wide = pre.merge(post, on=unit_col)
    wide["match_pair_id"] = np.arange(len(wide))
    wide["match_role"] = np.where(wide[treat_time_col].notna(), "treated", "control")
    wide["received_intervention"] = (wide["match_role"] == "treated").astype(int)
    return wide


def did_estimator(
    df: pd.DataFrame,
    method: DidMethod = "csdid",
    *,
    unit_col: str = "farm_id",
    time_col: str = "period",
    treat_time_col: str = "treatment_period",
    outcome_col: str = "yield",
    covariate_cols: Sequence[str] | None = None,
    n_boot: int = 999,
    alpha: float = 0.05,
    random_state: int = 42,
    **kwargs: object,
) -> DiDResult | ATTResult | BJSResult:
    """
    Route DiD estimation to pair-level, Callaway-Sant'Anna, or BJS imputation.

    Parameters
    ----------
    method:
        ``pair_diff`` — legacy matched pre/post DiD;
        ``csdid`` — Callaway & Sant'Anna (2021) staggered DR DiD;
        ``bjs`` — Borusyak, Jaravel & Spiess (2024) imputation;
        ``synthdid`` — Arkhangelsky et al. (2021) Synthetic DiD.
    """
    from analysis.bjs_imputation import BorusyakJaravelSpiess
    from analysis.csdid import CallawaySantAnna
    from analysis.synthdid import SyntheticDiD

    if method == "synthdid":
        est = SyntheticDiD(
            df,
            unit_col=unit_col,
            time_col=time_col,
            treat_time_col=treat_time_col,
            outcome_col=outcome_col,
            random_state=random_state,
        )
        res = est.estimate()
        return DiDResult(
            att=res.att,
            treated_change_mean=float("nan"),
            control_change_mean=float("nan"),
            n_pairs=res.n_treated,
            se=res.se,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            p_value=res.p_value,
            method="synthdid",
        )

    if method == "pair_diff":
        wide = df
        if time_col in df.columns and outcome_col in df.columns:
            wide = _long_to_matched_wide(
                df,
                unit_col=unit_col,
                time_col=time_col,
                outcome_col=outcome_col,
                treat_time_col=treat_time_col,
            )
        return calculate_did_att(
            wide,
            treat_time_col=treat_time_col,
            unit_col=unit_col,
            n_boot=int(kwargs.get("n_boot", n_boot)),  # type: ignore[arg-type]
            alpha=alpha,
            random_state=random_state,
        )

    if method == "csdid":
        est = CallawaySantAnna(
            df,
            unit_col=unit_col,
            time_col=time_col,
            treat_time_col=treat_time_col,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
            n_boot=n_boot,
            alpha=alpha,
            random_state=random_state,
        )
        res = est.simple_att()
        return DiDResult(
            att=res.att,
            treated_change_mean=float("nan"),
            control_change_mean=float("nan"),
            n_pairs=res.n_cells,
            se=res.se,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            p_value=None,
            method="csdid_simple_att",
        )

    if method == "bjs":
        est = BorusyakJaravelSpiess(
            df,
            unit_col=unit_col,
            time_col=time_col,
            treat_time_col=treat_time_col,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
            alpha=alpha,
            random_state=random_state,
        )
        return est.estimate()

    raise ValueError(f"Unknown did method: {method!r}")
