"""
Landscape-driven CSSVD incidence (Dumont et al. 2025).

Gradient-boosted Cox survival (scikit-survival ComponentwiseGradientBoostingSurvivalAnalysis,
Cox partial likelihood / CoxBoost-style) predicts hazard; 12-month incidence probability
with bootstrap prediction intervals.
"""

from __future__ import annotations

import structlog

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from data.cssvd_landscape_features import HORIZON_MONTHS, LandscapeFeatureRow, build_landscape_feature_row
from data.cssvd_strain_atlas import STRAIN_REGIONS, StrainRegion

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "cssvd_landscape.joblib"

NUMERIC_FEATURES: tuple[str, ...] = (
    "cocoa_probability_local",
    "non_cocoa_buffer_500m",
    "canopy_fragmentation_index",
    "extreme_precip_5day_count_yr",
    "dtr_growing_season",
)

STRAIN_PREFIX = "strain_"


@dataclass(frozen=True)
class IncidencePrediction:
    """12-month CSSVD incidence probability with 90% prediction interval."""

    point: float
    pi_low: float
    pi_high: float
    horizon_months: float = HORIZON_MONTHS


def _strain_dummies(strain: str) -> dict[str, float]:
    """One-hot strain (reference category ``2`` omitted)."""
    out = {f"{STRAIN_PREFIX}{r}": 0.0 for r in STRAIN_REGIONS if r != "2"}
    if strain in STRAIN_REGIONS and strain != "2":
        out[f"{STRAIN_PREFIX}{strain}"] = 1.0
    return out


def feature_dict_from_row(row: LandscapeFeatureRow | dict[str, Any]) -> dict[str, float]:
    """Flatten a landscape row to model feature keys."""
    if isinstance(row, LandscapeFeatureRow):
        base = row.to_dict()
    else:
        base = dict(row)
    strain = str(base.get("strain_region", "2"))
    feats = {k: float(base[k]) for k in NUMERIC_FEATURES if k in base}
    feats.update(_strain_dummies(strain))
    return feats


def features_to_dataframe(rows: list[dict[str, float]]) -> pd.DataFrame:
    cols = list(NUMERIC_FEATURES) + [f"{STRAIN_PREFIX}{r}" for r in STRAIN_REGIONS if r != "2"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].astype(np.float64)


def structured_survival_y(
    duration: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Build sksurv structured outcome array."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, duration, strict=True)],
        dtype=[("event", "?"), ("duration", "<f8")],
    )


def incidence_probability_at_horizon(
    model: ComponentwiseGradientBoostingSurvivalAnalysis,
    X: pd.DataFrame,
    *,
    horizon_months: float = HORIZON_MONTHS,
) -> np.ndarray:
    """
    P(incidence by horizon) = 1 - S(t) from predicted survival functions.
    """
    surv_fns = model.predict_survival_function(X)
    probs = np.zeros(len(X), dtype=np.float64)
    for i, fn in enumerate(surv_fns):
        times = fn.x
        surv = fn.y
        if len(times) == 0:
            probs[i] = 0.0
            continue
        s_at = float(np.interp(horizon_months, times, surv, left=1.0, right=float(surv[-1])))
        probs[i] = float(np.clip(1.0 - s_at, 0.0, 1.0))
    return probs


class LandscapeCSSVDModel:
    """
    CoxBoost-style survival model for 12-month CSSVD incidence.

    Training uses Dumont et al. supplement plots joined to landscape covariates.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
        horizon_months: float = HORIZON_MONTHS,
        n_bootstrap: int = 200,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.horizon_months = horizon_months
        self.n_bootstrap = n_bootstrap
        self._model: ComponentwiseGradientBoostingSurvivalAnalysis | None = None
        self._feature_columns: list[str] | None = None
        self._bootstrap_models: list[ComponentwiseGradientBoostingSurvivalAnalysis] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X: pd.DataFrame,
        duration: np.ndarray,
        event: np.ndarray,
        *,
        fit_bootstrap: bool = True,
    ) -> dict[str, float]:
        """
        Fit Cox gradient boosting on ``(duration, event)`` survival outcomes.

        Returns validation metrics dict (C-index on training rows).
        """
        y = structured_survival_y(duration, event)
        self._feature_columns = list(X.columns)
        model = ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            loss="coxph",
        )
        model.fit(X, y)
        self._model = model

        risk = model.predict(X)
        c_index, _, _, _, _ = concordance_index_censored(y["event"], y["duration"], risk)
        metrics = {"c_index_train": float(c_index)}

        if fit_bootstrap and self.n_bootstrap > 0:
            self._fit_bootstrap(X, y)

        return metrics

    def _fit_bootstrap(self, X: pd.DataFrame, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        models: list[ComponentwiseGradientBoostingSurvivalAnalysis] = []
        for b in range(self.n_bootstrap):
            for _attempt in range(20):
                idx = rng.integers(0, n, size=n)
                yb = y[idx]
                if bool(np.any(yb["event"])):
                    break
            else:
                continue
            Xb = X.iloc[idx].reset_index(drop=True)
            m = ComponentwiseGradientBoostingSurvivalAnalysis(
                n_estimators=max(20, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                random_state=self.random_state + b,
                loss="coxph",
            )
            try:
                m.fit(Xb, yb)
            except ValueError:
                continue
            models.append(m)
        self._bootstrap_models = models

    def _align_features(self, feats: dict[str, float]) -> pd.DataFrame:
        if self._feature_columns is None:
            cols = list(NUMERIC_FEATURES) + [f"{STRAIN_PREFIX}{r}" for r in STRAIN_REGIONS if r != "2"]
        else:
            cols = self._feature_columns
        row_df = features_to_dataframe([feats])
        return row_df[cols]

    def predict_from_features(self, feats: dict[str, float]) -> IncidencePrediction:
        if self._model is None:
            raise RuntimeError("LandscapeCSSVDModel is not fitted; call fit() or from_checkpoint()")
        X = self._align_features(feats)
        point = float(incidence_probability_at_horizon(self._model, X, horizon_months=self.horizon_months)[0])

        if self._bootstrap_models:
            boot_probs = []
            for m in self._bootstrap_models:
                boot_probs.append(
                    float(
                        incidence_probability_at_horizon(m, X, horizon_months=self.horizon_months)[0]
                    )
                )
            arr = np.asarray(boot_probs, dtype=np.float64)
            pi_low = float(np.percentile(arr, 5))
            pi_high = float(np.percentile(arr, 95))
        else:
            pi_low = point
            pi_high = point

        return IncidencePrediction(
            point=point,
            pi_low=pi_low,
            pi_high=pi_high,
            horizon_months=self.horizon_months,
        )

    def predict_12mo_incidence(
        self,
        lat: float,
        lon: float,
        year: int,
        *,
        use_gee_climate: bool = False,
    ) -> IncidencePrediction:
        """Build landscape features at ``(lat, lon)`` and predict 12-month incidence."""
        row = build_landscape_feature_row(
            lat, lon, year, use_gee_climate=use_gee_climate
        )
        return self.predict_from_features(feature_dict_from_row(row))

    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save unfitted model")
        payload = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "bootstrap_models": self._bootstrap_models,
            "horizon_months": self.horizon_months,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_bootstrap": self.n_bootstrap,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> LandscapeCSSVDModel:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSSVD landscape checkpoint not found: {path}")
        payload = joblib.load(path)
        inst = cls(
            n_estimators=int(payload.get("n_estimators", 100)),
            learning_rate=float(payload.get("learning_rate", 0.1)),
            random_state=int(payload.get("random_state", 42)),
            horizon_months=float(payload.get("horizon_months", HORIZON_MONTHS)),
            n_bootstrap=int(payload.get("n_bootstrap", 0)),
        )
        inst._model = payload["model"]
        inst._feature_columns = payload.get("feature_columns")
        inst._bootstrap_models = payload.get("bootstrap_models")
        return inst


@lru_cache(maxsize=512)
def _cached_landscape_predict(
    lat_r: float,
    lon_r: float,
    year: int,
    checkpoint: str,
) -> IncidencePrediction:
    model = LandscapeCSSVDModel.from_checkpoint(checkpoint)
    return model.predict_12mo_incidence(lat_r, lon_r, year)


def predict_with_cache(
    lat: float,
    lon: float,
    year: int,
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    grid_step: float = 0.05,
) -> IncidencePrediction:
    """LRU-cached prediction keyed by rounded lat/lon/year."""
    lat_r = round(lat / grid_step) * grid_step
    lon_r = round(lon / grid_step) * grid_step
    return _cached_landscape_predict(lat_r, lon_r, year, str(checkpoint))


def fit_synthetic_demo(*, n_samples: int = 400, random_state: int = 42) -> LandscapeCSSVDModel:
    """
    Fit a small model on synthetic data for tests/CI (no GEE, no supplement).
    """
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    duration: list[float] = []
    event: list[int] = []

    for _ in range(n_samples):
        non_cocoa = float(rng.uniform(0.2, 0.95))
        frag = float(rng.uniform(0.5, 2.5))
        extreme = int(rng.integers(0, 40))
        dtr = float(rng.uniform(4.0, 14.0))
        cocoa_p = float(rng.uniform(0.4, 0.99))
        strain = str(rng.choice(list(STRAIN_REGIONS)))
        feats = {
            "cocoa_probability_local": cocoa_p,
            "non_cocoa_buffer_500m": non_cocoa,
            "canopy_fragmentation_index": frag,
            "extreme_precip_5day_count_yr": float(extreme),
            "dtr_growing_season": dtr,
            **_strain_dummies(strain),
        }
        rows.append(feats)
        # Higher non-cocoa -> lower hazard (longer duration if no event)
        hazard = np.exp(
            -1.5 * non_cocoa
            + 0.05 * extreme
            + 0.08 * dtr
            - 0.3 * frag
            + 0.2 * (strain != "2")
        )
        dur = float(rng.exponential(1.0 / max(hazard, 0.05)) * 6 + 1)
        dur = min(dur, HORIZON_MONTHS * 2)
        ev = int(dur <= HORIZON_MONTHS and rng.random() < 0.35 * hazard)
        duration.append(dur)
        event.append(ev)

    X = features_to_dataframe(rows)
    y_dur = np.asarray(duration)
    y_ev = np.asarray(event)
    model = LandscapeCSSVDModel(n_estimators=30, n_bootstrap=20, random_state=random_state)
    model.fit(X, y_dur, y_ev, fit_bootstrap=True)
    return model


__all__ = [
    "DEFAULT_CHECKPOINT",
    "HORIZON_MONTHS",
    "IncidencePrediction",
    "LandscapeCSSVDModel",
    "feature_dict_from_row",
    "features_to_dataframe",
    "fit_synthetic_demo",
    "incidence_probability_at_horizon",
    "predict_with_cache",
    "structured_survival_y",
]
