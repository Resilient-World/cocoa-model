"""
Doubly-Valid/Doubly-Sharp (DVDS) sensitivity analysis under Tan's Marginal Sensitivity Model.

Reference: Dorn, Guo & Kallus (2022), arXiv:2112.11449 — sharp partial identification
bounds on the ATE with cross-fitted nuisances (propensity, quantile/CVaR, transformed outcome).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from analysis.psm_matching import _validate_psm_inputs

WALD_Z = 1.96
PS_CLIP = (0.01, 0.99)
HGB_KW = {"max_iter": 300, "learning_rate": 0.05}


@dataclass(frozen=True)
class MarginalSensitivityModel:
    """Tan (2006) marginal sensitivity model with odds-ratio bound Λ ≥ 1."""

    lambda_: float

    def __post_init__(self) -> None:
        if self.lambda_ < 1.0:
            raise ValueError("lambda_ must be >= 1")

    @property
    def tau(self) -> float:
        return self.lambda_ / (self.lambda_ + 1.0)


@dataclass
class DVDSResult:
    ate_lower: float
    ate_upper: float
    ate_ci_lower: float
    ate_ci_upper: float
    lambda_: float
    n: int
    nuisance_diagnostics: dict[str, Any] = field(default_factory=dict)


def _is_binary_outcome(y: np.ndarray) -> bool:
    uniq = np.unique(y[~np.isnan(y)])
    return len(uniq) <= 2 and set(np.round(uniq).astype(int)).issubset({0, 1})


def _lambda_sign_adj(
    y: np.ndarray,
    q: np.ndarray,
    lam: float,
    bound: Literal["plus", "minus"],
) -> np.ndarray:
    sign_yq = np.sign(y - q)
    sign_yq[sign_yq == 0] = 1
    if bound == "plus":
        exponent = sign_yq
    else:
        exponent = -sign_yq
    return q + (lam**exponent) * (y - q)


def _cvar_transform(y: np.ndarray, q: np.ndarray, lam: float, tau: float) -> np.ndarray:
    """Transformed outcome for ρ₊ regression (Eq. 18)."""
    return (1.0 / lam) * y + (1.0 - 1.0 / lam) * (q + np.maximum(y - q, 0.0) / (1.0 - tau))


def _binary_nuisances(mu: np.ndarray, lam: float, tau: float) -> tuple[np.ndarray, ...]:
    q_plus = (mu > 1.0 - tau).astype(float)
    q_minus = (mu > tau).astype(float)
    rho_plus = np.minimum(1.0 - 1.0 / lam + mu * lam, mu * lam)
    rho_minus = np.maximum(1.0 - lam + mu * lam, mu / lam)
    return q_plus, q_minus, rho_plus, rho_minus


def _phi_1(
    y: np.ndarray,
    z: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    lam: float,
    bound: Literal["plus", "minus"],
) -> np.ndarray:
    adj = _lambda_sign_adj(y, q, lam, bound)
    return z * y + (1.0 - z) * rho + ((1.0 - e) * z / e) * (adj - rho)


def _phi_0(
    y: np.ndarray,
    z: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    lam: float,
    bound: Literal["plus", "minus"],
) -> np.ndarray:
    adj = _lambda_sign_adj(y, q, lam, bound)
    return (1.0 - z) * y + z * rho + (e * (1.0 - z) / (1.0 - e)) * (adj - rho)


def _phi_ate(
    y: np.ndarray,
    z: np.ndarray,
    e: np.ndarray,
    q1_p: np.ndarray,
    q1_m: np.ndarray,
    q0_p: np.ndarray,
    q0_m: np.ndarray,
    r1_p: np.ndarray,
    r1_m: np.ndarray,
    r0_p: np.ndarray,
    r0_m: np.ndarray,
    lam: float,
    bound: Literal["plus", "minus"],
) -> np.ndarray:
    if bound == "plus":
        return _phi_1(y, z, e, q1_p, r1_p, lam, "plus") - _phi_0(y, z, e, q0_m, r0_m, lam, "minus")
    return _phi_1(y, z, e, q1_m, r1_m, lam, "minus") - _phi_0(y, z, e, q0_p, r0_p, lam, "plus")


def _fit_quantile(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    alpha: float,
    random_state: int,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha,
        random_state=random_state,
        **HGB_KW,
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def _fit_rho_regression(
    x_tr: np.ndarray,
    target_tr: np.ndarray,
    x_all: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(random_state=random_state, **HGB_KW)
    model.fit(x_tr, target_tr)
    return model.predict(x_all)


def _fit_fold_nuisances(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    a_tr: np.ndarray,
    x_all: np.ndarray,
    tr_idx: np.ndarray,
    *,
    lam: float,
    tau: float,
    binary: bool,
    random_state: int,
) -> dict[str, np.ndarray]:
    n = len(x_all)
    e = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)

    g = HistGradientBoostingClassifier(random_state=random_state, **HGB_KW)
    g.fit(x_tr, a_tr)
    e = np.clip(g.predict_proba(x_all)[:, 1], PS_CLIP[0], PS_CLIP[1])

    tr_t = a_tr == 1
    tr_c = a_tr == 0
    if tr_t.sum() > 0:
        m1 = HistGradientBoostingRegressor(random_state=random_state, **HGB_KW)
        m1.fit(x_tr[tr_t], y_tr[tr_t])
        mu1 = m1.predict(x_all)
    if tr_c.sum() > 0:
        m0 = HistGradientBoostingRegressor(random_state=random_state, **HGB_KW)
        m0.fit(x_tr[tr_c], y_tr[tr_c])
        mu0 = m0.predict(x_all)

    z_all = np.zeros(n)  # placeholder; caller passes z per row
    mu_z1 = mu1
    mu_z0 = mu0

    if binary:
        q1_p, q1_m, r1_p, r1_m = _binary_nuisances(mu_z1, lam, tau)
        q0_p, q0_m, r0_p, r0_m = _binary_nuisances(mu_z0, lam, tau)
        return {
            "e": e,
            "q1_plus": q1_p,
            "q1_minus": q1_m,
            "q0_plus": q0_p,
            "q0_minus": q0_m,
            "rho1_plus": r1_p,
            "rho1_minus": r1_m,
            "rho0_plus": r0_p,
            "rho0_minus": r0_m,
            "mu1_mean": float(mu1[tr_idx][a_tr == 1].mean()) if tr_t.sum() else 0.0,
            "mu0_mean": float(mu0[tr_idx][a_tr == 0].mean()) if tr_c.sum() else 0.0,
        }

    alpha_plus = 1.0 - tau
    alpha_minus = tau
    q_tau_z1 = np.zeros(n)
    q_tau_z0 = np.zeros(n)
    if tr_t.sum() > 0:
        q_tau_z1 = _fit_quantile(x_tr[tr_t], y_tr[tr_t], x_all, tau, random_state)
    if tr_c.sum() > 0:
        q_tau_z0 = _fit_quantile(x_tr[tr_c], y_tr[tr_c], x_all, tau, random_state)

    q1_plus = np.zeros(n)
    q1_minus = np.zeros(n)
    q0_plus = np.zeros(n)
    q0_minus = np.zeros(n)
    if tr_t.sum() > 0:
        q1_plus = _fit_quantile(x_tr[tr_t], y_tr[tr_t], x_all, alpha_plus, random_state)
        q1_minus = _fit_quantile(x_tr[tr_t], y_tr[tr_t], x_all, alpha_minus, random_state)
    if tr_c.sum() > 0:
        q0_plus = _fit_quantile(x_tr[tr_c], y_tr[tr_c], x_all, alpha_plus, random_state)
        q0_minus = _fit_quantile(x_tr[tr_c], y_tr[tr_c], x_all, alpha_minus, random_state)

    xz = np.column_stack([x_all, np.zeros(n)])
    x_tr_z1 = np.column_stack([x_tr, np.ones(len(x_tr))])
    x_tr_z0 = np.column_stack([x_tr, np.zeros(len(x_tr))])
    q1_tr_p = q1_plus[tr_idx]
    q1_tr_m = q1_minus[tr_idx]
    q0_tr_p = q0_plus[tr_idx]
    q0_tr_m = q0_minus[tr_idx]

    rho1_plus = np.zeros(n)
    rho1_minus = np.zeros(n)
    rho0_plus = np.zeros(n)
    rho0_minus = np.zeros(n)
    if tr_t.sum() > 0:
        t_plus = _cvar_transform(y_tr[tr_t], q1_tr_p[tr_t], lam, tau)
        t_minus = _cvar_transform(y_tr[tr_t], q1_tr_m[tr_t], lam, tau)
        rho1_plus = _fit_rho_regression(x_tr_z1[tr_t], t_plus, xz, random_state)
        rho1_minus = _fit_rho_regression(x_tr_z1[tr_t], t_minus, xz, random_state)
    if tr_c.sum() > 0:
        t0_plus = _cvar_transform(y_tr[tr_c], q0_tr_p[tr_c], lam, tau)
        t0_minus = _cvar_transform(y_tr[tr_c], q0_tr_m[tr_c], lam, tau)
        rho0_plus = _fit_rho_regression(x_tr_z0[tr_c], t0_plus, xz, random_state)
        rho0_minus = _fit_rho_regression(x_tr_z0[tr_c], t0_minus, xz, random_state)

    return {
        "e": e,
        "q1_plus": q1_plus,
        "q1_minus": q1_minus,
        "q0_plus": q0_plus,
        "q0_minus": q0_minus,
        "rho1_plus": rho1_plus,
        "rho1_minus": rho1_minus,
        "rho0_plus": rho0_plus,
        "rho0_minus": rho0_minus,
        "mu1_mean": float(mu_z1.mean()),
        "mu0_mean": float(mu_z0.mean()),
        "q_tau_z1_mean": float(q_tau_z1.mean()),
    }


def _wald_se(
    phi: np.ndarray,
    psi_hat: float,
    fold_ids: np.ndarray,
    n_folds: int,
    n: int,
) -> float:
    var_sum = 0.0
    for k in range(n_folds):
        mask = fold_ids == k
        if mask.sum() < 2:
            continue
        centered = phi[mask] - psi_hat
        var_sum += (mask.sum() / n) * float(np.var(centered, ddof=1))
    sigma2 = (n / max(n - 1, 1)) * var_sum
    return float(np.sqrt(max(sigma2, 0.0) / n))


def dvds_ate(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    lambda_: float,
    n_folds: int = 5,
    random_state: int = 42,
) -> DVDSResult:
    """
    DVDS point bounds and 95% Wald CIs for the sharp ATE partial identification set.

    Implements Algorithm 1 with Eq. (10)-(11) nuisances (binary closed form; continuous
    quantile + transformed-outcome regression).
    """
    msm = MarginalSensitivityModel(lambda_)
    lam = msm.lambda_
    tau = msm.tau

    cols = list(covariate_cols)
    _validate_psm_inputs(df, treatment_col, cols)
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome '{outcome_col}' not found")

    work = df.dropna(subset=[outcome_col, treatment_col, *cols]).copy()
    x = work[cols].to_numpy()
    z = work[treatment_col].to_numpy().astype(int)
    y = work[outcome_col].to_numpy().astype(float)
    n = len(y)
    if n < 10:
        raise ValueError(f"Need at least 10 rows for DVDS; got {n}")

    binary = _is_binary_outcome(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    phi_plus = np.zeros(n)
    phi_minus = np.zeros(n)
    fold_ids = np.zeros(n, dtype=int)
    fold_diag: list[dict[str, float]] = []

    for fold_k, (tr_idx, te_idx) in enumerate(skf.split(x, z)):
        fold_ids[te_idx] = fold_k
        nuis = _fit_fold_nuisances(
            x[tr_idx],
            y[tr_idx],
            z[tr_idx],
            x,
            tr_idx,
            lam=lam,
            tau=tau,
            binary=binary,
            random_state=random_state + fold_k,
        )
        fold_diag.append(
            {
                "fold": float(fold_k),
                "e_mean": float(nuis["e"].mean()),
                "mu1_mean": nuis.get("mu1_mean", 0.0),
                "mu0_mean": nuis.get("mu0_mean", 0.0),
            }
        )
        phi_plus[te_idx] = _phi_ate(
            y[te_idx],
            z[te_idx],
            nuis["e"][te_idx],
            nuis["q1_plus"][te_idx],
            nuis["q1_minus"][te_idx],
            nuis["q0_plus"][te_idx],
            nuis["q0_minus"][te_idx],
            nuis["rho1_plus"][te_idx],
            nuis["rho1_minus"][te_idx],
            nuis["rho0_plus"][te_idx],
            nuis["rho0_minus"][te_idx],
            lam,
            "plus",
        )
        phi_minus[te_idx] = _phi_ate(
            y[te_idx],
            z[te_idx],
            nuis["e"][te_idx],
            nuis["q1_plus"][te_idx],
            nuis["q1_minus"][te_idx],
            nuis["q0_plus"][te_idx],
            nuis["q0_minus"][te_idx],
            nuis["rho1_plus"][te_idx],
            nuis["rho1_minus"][te_idx],
            nuis["rho0_plus"][te_idx],
            nuis["rho0_minus"][te_idx],
            lam,
            "minus",
        )

    ate_upper = float(phi_plus.mean())
    ate_lower = float(phi_minus.mean())
    se_plus = _wald_se(phi_plus, ate_upper, fold_ids, n_folds, n)
    se_minus = _wald_se(phi_minus, ate_lower, fold_ids, n_folds, n)

    diagnostics: dict[str, Any] = {
        "outcome_type": "binary" if binary else "continuous",
        "tau": tau,
        "n_folds": n_folds,
        "n_treated": int(z.sum()),
        "folds": fold_diag,
    }

    return DVDSResult(
        ate_lower=ate_lower,
        ate_upper=ate_upper,
        ate_ci_lower=ate_lower - WALD_Z * se_minus,
        ate_ci_upper=ate_upper + WALD_Z * se_plus,
        lambda_=lam,
        n=n,
        nuisance_diagnostics=diagnostics,
    )


def tipping_point(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    lambda_min: float = 1.0,
    lambda_max: float = 10.0,
    tolerance: float = 1e-3,
    n_folds: int = 5,
    random_state: int = 42,
) -> float:
    """Smallest Λ in [lambda_min, lambda_max] where the 95% Wald partial-ID band contains 0."""

    def contains_zero(lam: float) -> bool:
        res = dvds_ate(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            covariate_cols=covariate_cols,
            lambda_=lam,
            n_folds=n_folds,
            random_state=random_state,
        )
        return res.ate_ci_lower <= 0.0 <= res.ate_ci_upper

    if contains_zero(lambda_min):
        return lambda_min
    if not contains_zero(lambda_max):
        return lambda_max

    lo, hi = lambda_min, lambda_max
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if contains_zero(mid):
            hi = mid
        else:
            lo = mid
    return hi


__all__ = [
    "DVDSResult",
    "MarginalSensitivityModel",
    "dvds_ate",
    "tipping_point",
]
