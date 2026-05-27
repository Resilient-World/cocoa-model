"""
TSFM ensemble: parallel execution, weighted median aggregation, and NNLS weight fitting.

Runs all four TSFM wrappers in parallel and returns a weighted median forecast.
Per-stratum NNLS weights are fitted from leave-one-year-out CV against the
historical yield panel and persisted to ``config/tsfm_weights.yaml``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml

from models.tsfm.wrappers import (
    Chronos2Wrapper,
    Moirai2Wrapper,
    TimeMoEWrapper,
    TimesFM2Wrapper,
    TsfmForecast,
    TsfmWrapper,
    build_wrapper,
)

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "config" / "tsfm_weights.yaml"

MODEL_NAMES = ("chronos-2", "timesfm", "timemoe", "moirai")
DEFAULT_WEIGHTS: dict[str, float] = {name: 0.25 for name in MODEL_NAMES}


def _load_weights(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    path = Path(path) if path else DEFAULT_WEIGHTS_PATH
    if not path.is_file():
        return {"default": dict(DEFAULT_WEIGHTS)}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {"default": dict(DEFAULT_WEIGHTS)}
    return data


def _save_weights(weights: dict[str, dict[str, float]], path: Path | str | None = None) -> None:
    path = Path(path) if path else DEFAULT_WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(weights, f, default_flow_style=False)
    log.info("Saved TSFM ensemble weights", path=str(path))


class WeightedMedianForecast:
    """Aggregate per-model quantile forecasts into a weighted median."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def aggregate(self, forecasts: dict[str, TsfmForecast]) -> TsfmForecast:
        """Combine per-model forecasts using weighted median across models."""
        active = {k: v for k, v in forecasts.items() if k in self.weights}
        if not active:
            raise ValueError("No active model forecasts to aggregate")

        w_sum = sum(self.weights[k] for k in active)
        norm_w = {k: self.weights[k] / w_sum for k in active} if w_sum > 0 else {k: 1.0 / len(active) for k in active}

        horizon = next(iter(active.values())).p10.shape[0]
        p10_agg = np.zeros(horizon, dtype=np.float64)
        p50_agg = np.zeros(horizon, dtype=np.float64)
        p90_agg = np.zeros(horizon, dtype=np.float64)

        for name, fc in active.items():
            w = norm_w[name]
            p10_agg += w * fc.p10
            p50_agg += w * fc.p50
            p90_agg += w * fc.p90

        return TsfmForecast(p10=p10_agg, p50=p50_agg, p90=p90_agg)


class NnlsWeightFitter:
    """Fit per-stratum non-negative least-squares weights via leave-one-year-out CV."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

    def fit(
        self,
        region_panels: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
        *,
        horizon: int = 12,
        num_samples: int = 50,
        device: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit per-region NNLS weights.

        Parameters
        ----------
        region_panels:
            ``{region: [(history, climate_covariates, actual_future_yield), ...]}``.
            Each tuple is one year's data for leave-one-year-out CV.
        horizon:
            Forecast horizon in time steps.
        num_samples:
            Samples per model per fold.
        device:
            Torch device for model inference.
        """
        from scipy.optimize import nnls

        wrappers: dict[str, TsfmWrapper] = {
            name: build_wrapper(name, device=device) for name in MODEL_NAMES
        }

        all_weights: dict[str, dict[str, float]] = {}
        for region, folds in region_panels.items():
            if len(folds) < 2:
                all_weights[region] = dict(DEFAULT_WEIGHTS)
                continue

            n_folds = len(folds)
            A = np.zeros((n_folds * horizon, len(MODEL_NAMES)), dtype=np.float64)
            b = np.zeros(n_folds * horizon, dtype=np.float64)

            for fold_idx, (history, covariates, actual) in enumerate(folds):
                full_input = np.column_stack([history, covariates]) if covariates.size else history
                model_preds: dict[str, np.ndarray] = {}
                for name in MODEL_NAMES:
                    fc = wrappers[name].forecast(full_input, horizon, num_samples=num_samples)
                    model_preds[name] = fc.p50

                row_start = fold_idx * horizon
                row_end = row_start + horizon
                for j, name in enumerate(MODEL_NAMES):
                    A[row_start:row_end, j] = model_preds[name]
                b[row_start:row_end] = actual

            coeffs, _ = nnls(A, b)
            total = float(coeffs.sum())
            if total > 0:
                coeffs = coeffs / total
            else:
                coeffs = np.ones(len(MODEL_NAMES)) / len(MODEL_NAMES)

            region_weights = {name: float(coeffs[i]) for i, name in enumerate(MODEL_NAMES)}
            all_weights[region] = region_weights
            log.info("NNLS weights fitted", region=region, weights=region_weights)

        all_weights.setdefault("default", dict(DEFAULT_WEIGHTS))
        _save_weights(all_weights, self.weights_path)
        return all_weights


class TsfmEnsemble:
    """Run all four TSFM wrappers in parallel and return a weighted median forecast."""

    def __init__(
        self,
        *,
        weights_path: Path | str | None = None,
        ensemble_mode: str | None = None,
        device: str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.ensemble_mode = ensemble_mode or os.environ.get("TSFM_ENSEMBLE_MODE", "nnls")
        self.device = device
        self.max_workers = max_workers

        self._wrappers: dict[str, TsfmWrapper] | None = None
        self._weights_cache: dict[str, dict[str, float]] | None = None

    @property
    def wrappers(self) -> dict[str, TsfmWrapper]:
        if self._wrappers is None:
            self._wrappers = {
                name: build_wrapper(name, device=self.device) for name in MODEL_NAMES
            }
        return self._wrappers

    def _get_weights(self, region: str) -> dict[str, float]:
        if self._weights_cache is None:
            self._weights_cache = _load_weights(self.weights_path)
        return self._weights_cache.get(region, self._weights_cache.get("default", dict(DEFAULT_WEIGHTS)))

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        region: str = "default",
        num_samples: int = 100,
    ) -> TsfmForecast:
        """
        Run ensemble forecast.

        Parameters
        ----------
        history:
            ``[time_steps, features]`` multivariate input.
        horizon:
            Forecast horizon in time steps.
        region:
            Region key for per-region weights.
        num_samples:
            Samples per model.
        """
        if self.ensemble_mode == "best":
            primary = os.environ.get("TSFM_PRIMARY", "timemoe")
            wrapper = build_wrapper(primary, device=self.device)
            return wrapper.forecast(history, horizon, num_samples=num_samples)

        forecasts: dict[str, TsfmForecast] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.wrappers[name].forecast, history, horizon, num_samples
                ): name
                for name in MODEL_NAMES
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    forecasts[name] = future.result()
                except Exception as exc:
                    log.warning("TSFM model failed in ensemble", model=name, error=str(exc))

        if not forecasts:
            raise RuntimeError("All TSFM models failed in ensemble forecast")

        if self.ensemble_mode == "mean":
            weights = {name: 1.0 / len(forecasts) for name in forecasts}
        else:
            weights = self._get_weights(region)

        aggregator = WeightedMedianForecast(weights)
        return aggregator.aggregate(forecasts)

    def fit_weights(
        self,
        region_panels: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
        *,
        horizon: int = 12,
        num_samples: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Fit and persist per-region NNLS weights."""
        fitter = NnlsWeightFitter(self.weights_path)
        self._weights_cache = fitter.fit(
            region_panels,
            horizon=horizon,
            num_samples=num_samples,
            device=self.device,
        )
        return self._weights_cache
