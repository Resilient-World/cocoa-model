"""
Propensity Score Matching (PSM) for causal impact evaluation (e.g. DiD).

Estimates propensity scores via logistic regression and performs 1:1 nearest-neighbor
matching on the propensity scale without replacement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_COVARIATES: tuple[str, ...] = (
    "farm_size_ha",
    "baseline_yield",
    "soil_quality_index",
    "historical_rainfall",
)


def _default_covariate_cols(df: pd.DataFrame, treatment_col: str, id_col: str) -> list[str]:
    exclude = {treatment_col, id_col}
    return [c for c in df.columns if c not in exclude]


def _validate_psm_inputs(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: Sequence[str],
) -> None:
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found in DataFrame")

    missing_cov = [c for c in covariate_cols if c not in df.columns]
    if missing_cov:
        raise ValueError(f"Covariate columns not found: {missing_cov}")

    y = df[treatment_col]
    if y.isna().any():
        raise ValueError(f"Missing values in treatment column '{treatment_col}'")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treatment_col}' must be binary (0/1), got values {sorted(y.unique())}"
        )

    if df[covariate_cols].isna().any().any():
        raise ValueError("Covariates contain missing values; impute or drop before PSM")

    n_treated = int((y == 1).sum())
    n_control = int((y == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Need at least one treated and one control unit (got {n_treated} treated, {n_control} control)"
        )


def compute_propensity_scores(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    covariate_cols: Sequence[str] | None = None,
    id_col: str = "farm_id",
    random_state: int = 42,
) -> pd.Series:
    """
    Fit logistic regression and return P(received_intervention | covariates).

    Parameters
    ----------
    df:
        Farm-level data with treatment indicator and covariates.
    treatment_col:
        Binary column (0 = control, 1 = treated).
    covariate_cols:
        Columns used to estimate propensity. Defaults to all columns except
        ``treatment_col`` and ``id_col``.
    id_col:
        Identifier column excluded from covariates when using defaults.
    random_state:
        Random seed for logistic regression.

    Returns
    -------
    pd.Series
        Propensity scores aligned with ``df.index``.
    """
    covariate_cols = list(
        covariate_cols if covariate_cols is not None else _default_covariate_cols(df, treatment_col, id_col)
    )
    _validate_psm_inputs(df, treatment_col, covariate_cols)

    x = df[covariate_cols].to_numpy()
    y = df[treatment_col].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x, y)
    scores = model.predict_proba(x)[:, 1]
    return pd.Series(scores, index=df.index, name="propensity_score")


def match_nearest_neighbor(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    ps_col: str = "propensity_score",
    caliper: float | None = None,
) -> pd.DataFrame:
    """
    1:1 nearest-neighbor propensity score matching without replacement.

    Parameters
    ----------
    df:
        DataFrame including treatment indicator and ``ps_col``.
    treatment_col:
        Binary treatment column (0/1).
    ps_col:
        Propensity score column.
    caliper:
        Maximum allowed |ps_treated - ps_control|. Treated units with no control
        within the caliper are dropped.

    Returns
    -------
    pd.DataFrame
        Matched sample only: two rows per pair (treated + control) with
        ``match_pair_id`` and ``match_role`` columns.
    """
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found")
    if ps_col not in df.columns:
        raise ValueError(f"Propensity column '{ps_col}' not found; run compute_propensity_scores first")

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    if treated.empty or control.empty:
        raise ValueError("Need at least one treated and one control unit to match")

    treated = treated.sort_values(ps_col, kind="mergesort")
    available_control_idx = list(control.index)
    control_ps = control.loc[available_control_idx, ps_col].to_numpy()

    pairs: list[tuple[object, object, int]] = []

    for t_idx, t_row in treated.iterrows():
        if not available_control_idx:
            break

        t_ps = t_row[ps_col]
        distances = np.abs(control_ps - t_ps)

        if caliper is not None:
            within = distances <= caliper
            if not within.any():
                continue
            candidate_positions = np.where(within)[0]
            candidate_distances = distances[within]
        else:
            candidate_positions = np.arange(len(distances))
            candidate_distances = distances

        best_local = int(candidate_positions[np.argmin(candidate_distances)])
        c_idx = available_control_idx[best_local]

        pair_id = len(pairs)
        pairs.append((t_idx, c_idx, pair_id))

        del available_control_idx[best_local]
        control_ps = np.delete(control_ps, best_local)

    if not pairs:
        raise ValueError(
            "No matched pairs found. Relax the caliper or check overlap in propensity scores."
        )

    matched_rows: list[pd.DataFrame] = []
    for t_idx, c_idx, pair_id in pairs:
        t_out = df.loc[[t_idx]].copy()
        c_out = df.loc[[c_idx]].copy()
        t_out["match_pair_id"] = pair_id
        c_out["match_pair_id"] = pair_id
        t_out["match_role"] = "treated"
        c_out["match_role"] = "control"
        matched_rows.append(t_out)
        matched_rows.append(c_out)

    return pd.concat(matched_rows, axis=0).reset_index(drop=True)


def propensity_score_match(
    df: pd.DataFrame,
    *,
    treatment_col: str = "received_intervention",
    covariate_cols: Sequence[str] | None = None,
    id_col: str = "farm_id",
    ps_col: str = "propensity_score",
    caliper: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    End-to-end PSM: propensity scores + nearest-neighbor matching.

    Returns a DataFrame containing only matched treated-control pairs,
    ready for difference-in-differences or other outcome analyses.
    """
    work = df.copy()
    work[ps_col] = compute_propensity_scores(
        work,
        treatment_col=treatment_col,
        covariate_cols=covariate_cols,
        id_col=id_col,
        random_state=random_state,
    )
    return match_nearest_neighbor(
        work,
        treatment_col=treatment_col,
        ps_col=ps_col,
        caliper=caliper,
    )
