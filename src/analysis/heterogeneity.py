"""
CATE estimation utilities (heterogeneous treatment effects).

- :class:`CausalForest` — honest causal forest via ``econml.dml.CausalForestDML``
  (Athey, Tibshirani & Wager 2019; Chernozhukov et al. 2018 cross-fitting).
- :class:`RLearnerCATE` — DR-forest learner via ``econml.dr.ForestDRLearner`` with
  HistGradientBoosting nuisances and honest forest final stage (Nie & Wager 2021).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

try:
    from econml.dml import CausalForestDML
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CATE estimation requires econml>=0.15. Install with: pip install 'econml>=0.15'"
    ) from exc

Method = Literal["r_learner", "causal_forest"]
Z_975 = 1.96


@dataclass(frozen=True)
class EffectResult:
    """Point CATE estimates with asymptotic intervals from econml inference."""

    point: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


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
    if not vals.issubset({0, 1}):
        raise ValueError("treatment must be binary 0/1")


def _as_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float64, copy=True)
    if np.isnan(x).any():
        raise ValueError("Covariates contain missing values; impute first")
    return x


def default_nuisance_models(
    random_state: int,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    """HGB nuisances aligned with AIPW / staggered DiD and DR policy learners."""
    return (
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
        ),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
        ),
    )


def _resolve_n_estimators(n_estimators: int | None) -> int:
    env = os.environ.get("CATE_N_ESTIMATORS")
    if n_estimators is not None:
        n = int(n_estimators)
    elif env:
        n = int(env)
    else:
        n = 2000
    # CausalForestDML requires n_estimators divisible by subforest_size (default 4)
    if n % 4 != 0:
        n = n + (4 - n % 4)
    return n


def _build_causal_forest_dml(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    cv: int,
    random_state: int,
) -> CausalForestDML:
    model_y, model_t = default_nuisance_models(random_state)
    return CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=cv,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        honest=True,
        inference=True,
        n_jobs=-1,
        random_state=random_state,
    )


class _BaseCATEEstimator:
    """Shared fit / effect API for econml-backed CATE estimators."""

    _estimator: Any
    covariate_cols_: list[str] | None = None
    X_train_: np.ndarray | None = None

    def _extract_arrays(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _check_binary_treatment(df[treatment_col].astype(int))
        y = df[outcome_col].astype(float).to_numpy()
        t = df[treatment_col].astype(int).to_numpy()
        x = _as_float_array(df, covariate_cols)
        return y, t, x

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> _BaseCATEEstimator:
        y, t, x = self._extract_arrays(df, treatment_col, outcome_col, covariate_cols)
        self.covariate_cols_ = list(covariate_cols)
        self.X_train_ = x
        self._estimator.fit(y, t, X=x)
        return self

    def fit_arrays(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> _BaseCATEEstimator:
        """Numpy API for backward compatibility."""
        work = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        work["__y"] = y
        work["__t"] = t
        return self.fit(work, "__t", "__y", list(work.columns[:-2]))

    def effect(self, X_new: np.ndarray, *, alpha: float = 0.05) -> EffectResult:
        lo, hi = self.effect_interval(X_new, alpha=alpha)
        point = np.asarray(self._estimator.effect(X_new), dtype=np.float64).ravel()
        lo = np.asarray(lo, dtype=np.float64).ravel()
        hi = np.asarray(hi, dtype=np.float64).ravel()
        z = Z_975 if alpha == 0.05 else 1.96
        se = (hi - lo) / (2.0 * z)
        return EffectResult(point=point, se=se, ci_low=lo, ci_high=hi)

    def effect_interval(
        self,
        X_new: np.ndarray,
        *,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        lb, ub = self._estimator.effect_interval(X_new, alpha=alpha)
        return np.asarray(lb, dtype=np.float64).ravel(), np.asarray(ub, dtype=np.float64).ravel()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.effect(x).point.astype(np.float32)

    def feature_importances(
        self,
        *,
        include_permutation: bool = False,
        include_shap: bool = False,
    ) -> pd.DataFrame:
        if self.covariate_cols_ is None:
            raise ValueError("Estimator not fit")
        cols = self.covariate_cols_
        split_imp = np.zeros(len(cols), dtype=np.float64)
        est = self._estimator
        if hasattr(est, "feature_importances"):
            try:
                fi = est.feature_importances()
                split_imp = np.asarray(fi, dtype=np.float64).ravel()[: len(cols)]
            except Exception:
                pass
        elif hasattr(est, "model_final_") and hasattr(est.model_final_, "feature_importances"):
            try:
                fi = est.model_final_.feature_importances()
                split_imp = np.asarray(fi, dtype=np.float64).ravel()[: len(cols)]
            except Exception:
                pass
        elif hasattr(est, "feature_importances_"):
            split_imp = np.asarray(est.feature_importances_, dtype=np.float64).ravel()[: len(cols)]

        perm_imp = np.full(len(cols), np.nan)
        if include_permutation and self.X_train_ is not None and len(self.X_train_) >= 20:
            y_proxy = self.effect(self.X_train_).point
            try:
                pi = permutation_importance(
                    self,
                    self.X_train_,
                    y_proxy,
                    n_repeats=3,
                    random_state=0,
                    n_jobs=1,
                )
                perm_imp = pi.importances_mean
            except Exception:
                pass

        shap_imp = np.full(len(cols), np.nan)
        if include_shap:
            try:
                import shap

                if self.X_train_ is not None and hasattr(est, "effect"):
                    explainer = shap.Explainer(
                        lambda data: self.effect(np.asarray(data)).point,
                        self.X_train_,
                    )
                    sv = explainer(self.X_train_[: min(200, len(self.X_train_))])
                    shap_imp = np.abs(np.asarray(sv.values)).mean(axis=0).ravel()[: len(cols)]
            except Exception:
                pass

        out = pd.DataFrame(
            {
                "feature": cols,
                "split_importance": split_imp,
                "permutation_importance": perm_imp,
                "shap_importance": shap_imp,
            }
        )
        return out.sort_values("split_importance", ascending=False).reset_index(drop=True)


def _fit_dispatch(
    est: _BaseCATEEstimator,
    arg1: pd.DataFrame | np.ndarray,
    arg2: str | np.ndarray,
    arg3: str | np.ndarray,
    arg4: list[str] | np.ndarray | None = None,
) -> _BaseCATEEstimator:
    """Support DataFrame API and legacy ``fit(x, y, t)`` numpy API."""
    if isinstance(arg1, pd.DataFrame):
        if not isinstance(arg2, str) or not isinstance(arg3, str) or not isinstance(arg4, list):
            raise TypeError("fit(df, treatment_col, outcome_col, covariate_cols) expected")
        return est.fit(arg1, arg2, arg3, arg4)
    if (
        isinstance(arg1, np.ndarray)
        and isinstance(arg2, np.ndarray)
        and isinstance(arg3, np.ndarray)
    ):
        return est.fit_arrays(arg1, arg2, arg3)
    raise TypeError("fit requires (df, treatment_col, outcome_col, covariate_cols) or (x, y, t)")


class CausalForest(_BaseCATEEstimator):
    """
    Honest causal forest CATE via ``econml.dml.CausalForestDML``.

    References: Athey, Tibshirani & Wager (2019); Chernozhukov et al. (2018).
    """

    def __init__(
        self,
        *,
        n_folds: int = 5,
        random_state: int = 42,
        n_estimators: int | None = None,
        min_samples_leaf: int = 10,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        n_est = _resolve_n_estimators(n_estimators)
        self._estimator = _build_causal_forest_dml(
            n_estimators=n_est,
            min_samples_leaf=min_samples_leaf,
            cv=n_folds,
            random_state=random_state,
        )

    def fit(
        self,
        arg1: pd.DataFrame | np.ndarray,
        arg2: str | np.ndarray,
        arg3: str | np.ndarray,
        arg4: list[str] | None = None,
    ) -> CausalForest:
        if isinstance(arg1, np.ndarray):
            return _fit_dispatch(self, arg1, arg2, arg3)  # type: ignore[arg-type]
        if arg4 is None:
            raise TypeError("covariate_cols required when fitting from DataFrame")
        return super().fit(arg1, arg2, arg3, arg4)


class RLearnerCATE(_BaseCATEEstimator):
    """
    R-learner CATE with honest forest final stage (Nie & Wager 2021).

    Implemented via ``econml.dml.CausalForestDML`` (extends econml's ``_RLearner`` with
    ``CausalForestDML`` as the final CATE regressor). ``metalearners.RLearner`` is not
    exported in econml 0.16+.
    """

    def __init__(
        self,
        *,
        n_folds: int = 5,
        random_state: int = 42,
        n_estimators: int | None = None,
        min_samples_leaf: int = 10,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        n_est = _resolve_n_estimators(n_estimators)
        self._estimator = _build_causal_forest_dml(
            n_estimators=n_est,
            min_samples_leaf=min_samples_leaf,
            cv=n_folds,
            random_state=random_state,
        )

    def fit(
        self,
        arg1: pd.DataFrame | np.ndarray,
        arg2: str | np.ndarray,
        arg3: str | np.ndarray,
        arg4: list[str] | None = None,
    ) -> RLearnerCATE:
        if isinstance(arg1, np.ndarray):
            return _fit_dispatch(self, arg1, arg2, arg3)  # type: ignore[arg-type]
        if arg4 is None:
            raise TypeError("covariate_cols required when fitting from DataFrame")
        return super().fit(arg1, arg2, arg3, arg4)


# Backward-compatible alias
RLearner = RLearnerCATE


def estimate_cate(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: list[str],
    method: Method = "r_learner",
    n_folds: int = 5,
    random_state: int = 42,
    n_estimators: int | None = None,
) -> CATEResult:
    """Estimate CATE with honest intervals; dispatches to :class:`CausalForest` or :class:`RLearnerCATE`."""
    if outcome not in df.columns or treatment not in df.columns:
        raise ValueError("Missing outcome/treatment columns")
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariate columns: {missing}")

    if method == "r_learner":
        est: _BaseCATEEstimator = RLearnerCATE(
            n_folds=n_folds,
            random_state=random_state,
            n_estimators=n_estimators,
        )
    elif method == "causal_forest":
        est = CausalForest(
            n_folds=n_folds,
            random_state=random_state,
            n_estimators=n_estimators,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    est.fit(df, treatment, outcome, covariates)
    x = _as_float_array(df, covariates)
    eff = est.effect(x, alpha=0.05)

    fi_df = est.feature_importances()
    fi = None
    if not fi_df.empty and "split_importance" in fi_df.columns:
        fi = pd.Series(
            fi_df["split_importance"].to_numpy(),
            index=fi_df["feature"].tolist(),
        ).sort_values(ascending=False)

    idx = df.index
    return CATEResult(
        tau_hat=pd.Series(eff.point, index=idx, name="tau_hat"),
        se=pd.Series(eff.se, index=idx, name="se"),
        ci_low=pd.Series(eff.ci_low, index=idx, name="ci_low"),
        ci_high=pd.Series(eff.ci_high, index=idx, name="ci_high"),
        feature_importances=fi,
        method=method,
        n_folds=n_folds,
    )


__all__ = [
    "CATEResult",
    "CausalForest",
    "EffectResult",
    "Method",
    "RLearner",
    "RLearnerCATE",
    "default_nuisance_models",
    "estimate_cate",
]
