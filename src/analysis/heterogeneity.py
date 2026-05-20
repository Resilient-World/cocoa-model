"""
CATE estimation utilities (heterogeneous treatment effects).

Implements:
- R-learner (Nie & Wager 2021) with cross-fitted nuisance models
- A lightweight tree-ensemble variant ("CausalForest") based on ExtraTrees over R-learner pseudo-outcomes

This module is designed to be dependency-light (scikit-learn + pandas/numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

Method = Literal["r_learner", "causal_forest"]


@dataclass(frozen=True)
class CATEResult:
    tau_hat: pd.Series
    se: pd.Series
    ci_low: pd.Series
    ci_high: pd.Series
    feature_importances: pd.Series | None = None
    method: str = "r_learner"
    n_folds: int = 5


def _check_binary_treatment(t: pd.Series) -> None:
    vals = set(pd.unique(t.dropna()))
    if not vals.issubset({0, 1, False, True}):
        raise ValueError("treatment must be binary 0/1")


def _as_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float32, copy=True)
    if np.isnan(x).any():
        raise ValueError("Covariates contain missing values; impute first")
    return x


def _crossfit_nuisances(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    outcome_model: Any,
    treatment_model: Any,
    n_folds: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cross-fitted m(x)=E[Y|X] and e(x)=P[T=1|X]."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat = np.zeros_like(t, dtype=np.float32)
    for tr, te in kf.split(x):
        m = clone(outcome_model)
        g = clone(treatment_model)
        m.fit(x[tr], y[tr])
        m_hat[te] = m.predict(x[te]).astype(np.float32)

        g.fit(x[tr], t[tr])
        if hasattr(g, "predict_proba"):
            e_hat[te] = g.predict_proba(x[te])[:, 1].astype(np.float32)
        else:
            e_hat[te] = g.predict(x[te]).astype(np.float32)
    e_hat = np.clip(e_hat, 1e-3, 1.0 - 1e-3)
    return m_hat, e_hat


class RLearner:
    """
    R-learner for conditional average treatment effects (Nie & Wager 2021).

    Minimizes: E[( (Y - m(X)) - tau(X) * (T - e(X)) )^2]

    We fit tau(X) by regressing pseudo-outcome z = (Y - m) / (T - e)
    with weights w = (T - e)^2 (stabilized), using cross-fitted nuisances.
    """

    def __init__(
        self,
        outcome_model: Any | None = None,
        treatment_model: Any | None = None,
        final_model: Any | None = None,
        *,
        n_folds: int = 5,
        random_state: int = 42,
        eps: float = 1e-3,
    ) -> None:
        self.outcome_model = outcome_model or HistGradientBoostingRegressor(
            random_state=random_state, max_depth=3, max_iter=30
        )
        self.treatment_model = treatment_model or HistGradientBoostingClassifier(
            random_state=random_state, max_depth=3, max_iter=30
        )
        self.final_model = final_model or HistGradientBoostingRegressor(
            random_state=random_state, max_depth=3, max_iter=30
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.eps = eps
        self.model_: RegressorMixin | BaseEstimator | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> "RLearner":
        m_hat, e_hat = _crossfit_nuisances(
            x,
            y,
            t,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        y_res = y - m_hat
        t_res = t - e_hat
        z = y_res / (t_res + np.sign(t_res) * self.eps + (t_res == 0) * self.eps)
        w = np.clip(t_res**2, self.eps, None)
        model = clone(self.final_model)
        model.fit(x, z, sample_weight=w)
        self.model_ = model
        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = np.asarray(getattr(model, "feature_importances_"))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("RLearner not fit")
        return np.asarray(self.model_.predict(x), dtype=np.float32)


class CausalForest:
    """
    Lightweight causal-forest-style estimator.

    This is not a full honest causal forest; it trains an ExtraTreesRegressor on the
    same R-learner pseudo-outcome with weights to capture non-linear heterogeneity.
    """

    def __init__(
        self,
        *,
        outcome_model: Any | None = None,
        treatment_model: Any | None = None,
        final_model: Any | None = None,
        n_folds: int = 5,
        random_state: int = 42,
        eps: float = 1e-3,
        n_estimators: int = 60,
        min_samples_leaf: int = 60,
    ) -> None:
        self.outcome_model = outcome_model or HistGradientBoostingRegressor(
            random_state=random_state, max_depth=3, max_iter=30
        )
        self.treatment_model = treatment_model or HistGradientBoostingClassifier(
            random_state=random_state, max_depth=3, max_iter=30
        )
        self.final_model = final_model or ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            max_features=0.7,
            n_jobs=-1,
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.eps = eps
        self.model_: Any | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> "CausalForest":
        m_hat, e_hat = _crossfit_nuisances(
            x,
            y,
            t,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        y_res = y - m_hat
        t_res = t - e_hat
        z = y_res / (t_res + np.sign(t_res) * self.eps + (t_res == 0) * self.eps)
        w = np.clip(t_res**2, self.eps, None)
        model = clone(self.final_model)
        model.fit(x, z, sample_weight=w)
        self.model_ = model
        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = np.asarray(getattr(model, "feature_importances_"))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("CausalForest not fit")
        return np.asarray(self.model_.predict(x), dtype=np.float32)


def estimate_cate(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: list[str],
    method: Method = "r_learner",
    n_folds: int = 5,
    random_state: int = 42,
) -> CATEResult:
    if outcome not in df.columns or treatment not in df.columns:
        raise ValueError("Missing outcome/treatment columns")
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariate columns: {missing}")

    y_s = df[outcome].astype(float)
    t_s = df[treatment].astype(int)
    _check_binary_treatment(t_s)

    x = _as_float_array(df, covariates)
    y = y_s.to_numpy(dtype=np.float32, copy=True)
    t = t_s.to_numpy(dtype=np.float32, copy=True)

    if method == "r_learner":
        est: Any = RLearner(n_folds=n_folds, random_state=random_state)
    elif method == "causal_forest":
        est = CausalForest(n_folds=n_folds, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    est.fit(x, y, t)
    tau = est.predict(x)

    # Simple, conservative uncertainty proxy:
    # use global residual scale from weighted objective; per-point SE scales with 1/sqrt(w).
    # (Good enough for ranking and unit tests; not a formal IF-based SE.)
    # Recompute residuals with in-sample tau.
    m_hat, e_hat = _crossfit_nuisances(
        x,
        y,
        t,
        outcome_model=est.outcome_model,
        treatment_model=est.treatment_model,
        n_folds=n_folds,
        random_state=random_state,
    )
    y_res = y - m_hat
    t_res = t - e_hat
    w = np.clip(t_res**2, 1e-3, None)
    resid = y_res - tau * t_res
    sigma2 = float(np.average(resid**2, weights=w))
    se = np.sqrt(sigma2 / w).astype(np.float32)
    ci_low = tau - 1.96 * se
    ci_high = tau + 1.96 * se

    fi = None
    if getattr(est, "feature_importances_", None) is not None:
        fi_arr = np.asarray(est.feature_importances_, dtype=np.float32)
        if fi_arr.shape[0] == len(covariates):
            fi = pd.Series(fi_arr, index=covariates).sort_values(ascending=False)

    idx = df.index
    return CATEResult(
        tau_hat=pd.Series(tau, index=idx, name="tau_hat"),
        se=pd.Series(se, index=idx, name="se"),
        ci_low=pd.Series(ci_low, index=idx, name="ci_low"),
        ci_high=pd.Series(ci_high, index=idx, name="ci_high"),
        feature_importances=fi,
        method=method,
        n_folds=n_folds,
    )

