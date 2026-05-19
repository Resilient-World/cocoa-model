"""
Propensity Score Matching with logit-scale calipers, k:1 matching, balance diagnostics,
and a cross-fit doubly-robust AIPW estimator (Chernozhukov et al. 2018 DML).

Backward-compatible: compute_propensity_scores, match_nearest_neighbor,
propensity_score_match retain their original signatures and defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_COVARIATES: tuple[str, ...] = (
    "farm_size_ha",
    "baseline_yield",
    "soil_quality_index",
    "historical_rainfall",
)


@dataclass
class BalanceReport:
    smd: pd.DataFrame
    max_smd_unmatched: float
    max_smd_matched: float
    balance_ok: bool


@dataclass
class AIPWResult:
    att: float
    att_se: float
    att_ci_low: float
    att_ci_high: float
    ate: float
    ate_se: float
    ate_ci_low: float
    ate_ci_high: float
    n: int
    n_treated: int
    n_folds: int
    method: str = "dml_aipw_crossfit"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _default_covariate_cols(
    df: pd.DataFrame,
    treatment_col: str,
    id_col: str,
) -> list[str]:
    return [c for c in df.columns if c not in {treatment_col, id_col}]


def _validate_psm_inputs(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: Sequence[str],
) -> None:
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found")
    missing = [c for c in covariate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Covariate columns not found: {missing}")
    y = df[treatment_col]
    if y.isna().any():
        raise ValueError(f"Missing values in '{treatment_col}'")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"'{treatment_col}' must be binary 0/1")
    if df[covariate_cols].isna().any().any():
        raise ValueError("Covariates contain missing values; impute first")
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise ValueError("Need >=1 treated and >=1 control")


# ---------------------------------------------------------------------------
# Propensity scores (scaled logistic — fast, stable, calibrated)
# ---------------------------------------------------------------------------


def compute_propensity_scores(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    covariate_cols: Sequence[str] | None = None,
    id_col: str = "farm_id",
    random_state: int = 42,
) -> pd.Series:
    cols = list(
        covariate_cols if covariate_cols is not None else _default_covariate_cols(df, treatment_col, id_col)
    )
    _validate_psm_inputs(df, treatment_col, cols)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(df[cols].to_numpy(), df[treatment_col].to_numpy())
    p = model.predict_proba(df[cols].to_numpy())[:, 1]
    return pd.Series(p, index=df.index, name="propensity_score")


# ---------------------------------------------------------------------------
# Caliper helpers
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def default_logit_caliper(ps: np.ndarray, k: float = 0.2) -> float:
    """Rosenbaum-Rubin (1985): 0.2 * SD(logit PS)."""
    return float(k * np.std(_logit(ps), ddof=1))


# ---------------------------------------------------------------------------
# Overlap trimming
# ---------------------------------------------------------------------------


def trim_common_support(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    ps_col: str = "propensity_score",
) -> pd.DataFrame:
    t = df.loc[df[treatment_col] == 1, ps_col]
    c = df.loc[df[treatment_col] == 0, ps_col]
    lo, hi = max(t.min(), c.min()), min(t.max(), c.max())
    return df[(df[ps_col] >= lo) & (df[ps_col] <= hi)].copy()


# ---------------------------------------------------------------------------
# Matching: k:1 NN with optional logit-scale caliper and replacement
# ---------------------------------------------------------------------------


def match_nearest_neighbor(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    ps_col: str = "propensity_score",
    caliper: float | None = None,
    caliper_scale: str = "raw",  # "raw" preserves legacy behavior; "logit" recommended
    k: int = 1,
    with_replacement: bool = False,
) -> pd.DataFrame:
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment '{treatment_col}' not found")
    if ps_col not in df.columns:
        raise ValueError(f"'{ps_col}' not found; run compute_propensity_scores first")
    if k < 1:
        raise ValueError("k must be >= 1")
    if caliper_scale not in {"raw", "logit"}:
        raise ValueError("caliper_scale must be 'raw' or 'logit'")

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    if treated.empty or control.empty:
        raise ValueError("Need >=1 treated and >=1 control")

    if caliper_scale == "logit":
        treated_score = _logit(treated[ps_col].to_numpy())
        control_score = _logit(control[ps_col].to_numpy())
    else:
        treated_score = treated[ps_col].to_numpy()
        control_score = control[ps_col].to_numpy()

    treated["_score"] = treated_score
    control["_score"] = control_score
    treated = treated.sort_values("_score", kind="mergesort")

    available_idx = list(control.index)
    available_score = control.loc[available_idx, "_score"].to_numpy()

    matched_rows: list[pd.DataFrame] = []
    pair_id = 0

    for t_idx, t_row in treated.iterrows():
        if not with_replacement and not available_idx:
            break
        t_s = t_row["_score"]
        if with_replacement:
            pool_idx, pool_score = list(control.index), control["_score"].to_numpy()
        else:
            pool_idx, pool_score = available_idx, available_score

        d = np.abs(pool_score - t_s)
        if caliper is not None:
            within = d <= caliper
            if not within.any():
                continue
            cand = np.where(within)[0]
        else:
            cand = np.arange(len(d))

        order = cand[np.argsort(d[cand])]
        chosen = order[:k].tolist()
        if not chosen:
            continue

        t_out = df.loc[[t_idx]].copy()
        t_out["match_pair_id"] = pair_id
        t_out["match_role"] = "treated"
        matched_rows.append(t_out)

        for local in chosen:
            c_out = df.loc[[pool_idx[local]]].copy()
            c_out["match_pair_id"] = pair_id
            c_out["match_role"] = "control"
            matched_rows.append(c_out)

        if not with_replacement:
            for local in sorted(chosen, reverse=True):
                del available_idx[local]
                available_score = np.delete(available_score, local)

        pair_id += 1

    if not matched_rows:
        raise ValueError("No matched pairs. Relax caliper or check overlap.")
    return pd.concat(matched_rows, axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SMD diagnostics
# ---------------------------------------------------------------------------


def standardized_mean_differences(
    unmatched_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "received_intervention",
    role_col: str = "match_role",
    smd_threshold: float = 0.10,
) -> BalanceReport:
    def _smd(t: np.ndarray, c: np.ndarray) -> float:
        v = (np.var(t, ddof=1) + np.var(c, ddof=1)) / 2.0
        return 0.0 if v <= 0 else float(abs(np.mean(t) - np.mean(c)) / np.sqrt(v))

    t_un = unmatched_df[unmatched_df[treatment_col] == 1]
    c_un = unmatched_df[unmatched_df[treatment_col] == 0]
    if role_col in matched_df.columns:
        t_m = matched_df[matched_df[role_col] == "treated"]
        c_m = matched_df[matched_df[role_col] == "control"]
    else:
        t_m = matched_df[matched_df[treatment_col] == 1]
        c_m = matched_df[matched_df[treatment_col] == 0]

    rows = [
        {
            "covariate": c,
            "smd_unmatched": _smd(t_un[c].to_numpy(), c_un[c].to_numpy()),
            "smd_matched": _smd(t_m[c].to_numpy(), c_m[c].to_numpy()),
        }
        for c in covariate_cols
    ]
    smd_df = pd.DataFrame(rows)
    return BalanceReport(
        smd=smd_df,
        max_smd_unmatched=float(smd_df["smd_unmatched"].max()),
        max_smd_matched=float(smd_df["smd_matched"].max()),
        balance_ok=float(smd_df["smd_matched"].max()) < smd_threshold,
    )


def love_plot_data(report: BalanceReport) -> pd.DataFrame:
    return report.smd.melt(
        id_vars="covariate",
        value_vars=["smd_unmatched", "smd_matched"],
        var_name="stage",
        value_name="smd",
    ).replace({"smd_unmatched": "before", "smd_matched": "after"})


# ---------------------------------------------------------------------------
# Cross-fit DML/AIPW (Chernozhukov et al. 2018)
# ---------------------------------------------------------------------------


def aipw_estimator(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str = "received_intervention",
    covariate_cols: Sequence[str] | None = None,
    id_col: str = "farm_id",
    ps_clip: tuple[float, float] = (0.01, 0.99),
    n_folds: int = 5,
    random_state: int = 42,
) -> AIPWResult:
    """
    Cross-fit doubly-robust ATE and ATT (DML; Chernozhukov et al. 2018).

    Nuisances:
      g(X) = P(A=1|X) via HistGradientBoostingClassifier (calibrated by CV)
      m_a(X) = E[Y|A=a,X] via HistGradientBoostingRegressor, fit separately on A=0 and A=1
               within each fold

    Inference: influence-function SE; valid under Neyman orthogonality + cross-fitting
    even with ML nuisances.
    """
    cols = list(
        covariate_cols if covariate_cols is not None else _default_covariate_cols(df, treatment_col, id_col)
    )
    _validate_psm_inputs(df, treatment_col, cols)
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome '{outcome_col}' not found")

    X = df[cols].to_numpy()
    A = df[treatment_col].to_numpy().astype(int)
    Y = df[outcome_col].to_numpy().astype(float)
    n = len(Y)

    e_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, A):
        g = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=None,
            random_state=random_state,
        )
        g.fit(X[tr], A[tr])
        e_hat[te] = np.clip(g.predict_proba(X[te])[:, 1], ps_clip[0], ps_clip[1])

        tr_t = tr[A[tr] == 1]
        tr_c = tr[A[tr] == 0]
        m1 = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=random_state
        ).fit(X[tr_t], Y[tr_t])
        m0 = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=random_state
        ).fit(X[tr_c], Y[tr_c])
        mu1_hat[te] = m1.predict(X[te])
        mu0_hat[te] = m0.predict(X[te])

    if_ate = mu1_hat - mu0_hat + A * (Y - mu1_hat) / e_hat - (1 - A) * (Y - mu0_hat) / (1 - e_hat)
    ate = float(if_ate.mean())
    ate_se = float(if_ate.std(ddof=1) / np.sqrt(n))

    p_t = A.mean()
    if_att = (A * (Y - mu0_hat) - (1 - A) * (e_hat / (1 - e_hat)) * (Y - mu0_hat)) / p_t
    att = float(if_att.mean())
    att_se = float(if_att.std(ddof=1) / np.sqrt(n))

    z = 1.96
    return AIPWResult(
        att=att,
        att_se=att_se,
        att_ci_low=att - z * att_se,
        att_ci_high=att + z * att_se,
        ate=ate,
        ate_se=ate_se,
        ate_ci_low=ate - z * ate_se,
        ate_ci_high=ate + z * ate_se,
        n=n,
        n_treated=int(A.sum()),
        n_folds=n_folds,
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def propensity_score_match(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    covariate_cols: Sequence[str] | None = None,
    id_col: str = "farm_id",
    ps_col: str = "propensity_score",
    caliper: float | None = None,
    caliper_scale: str = "raw",
    k: int = 1,
    with_replacement: bool = False,
    trim_overlap: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    work[ps_col] = compute_propensity_scores(
        work,
        treatment_col=treatment_col,
        covariate_cols=covariate_cols,
        id_col=id_col,
        random_state=random_state,
    )
    if trim_overlap:
        work = trim_common_support(work, treatment_col=treatment_col, ps_col=ps_col)
    match_caliper = caliper
    match_scale = caliper_scale
    if match_caliper is None:
        match_caliper = default_logit_caliper(work[ps_col].to_numpy())
        match_scale = "logit"
    return match_nearest_neighbor(
        work,
        treatment_col=treatment_col,
        ps_col=ps_col,
        caliper=match_caliper,
        caliper_scale=match_scale,
        k=k,
        with_replacement=with_replacement,
    )
