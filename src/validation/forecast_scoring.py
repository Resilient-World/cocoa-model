"""
Probabilistic forecast scoring: CRPS, PIT, reliability, sharpness, Energy Score.

Wrappers around properscoring (Apache-2.0) and scoringrules (MIT).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import properscoring as ps
import scoringrules as sr
from scipy import stats

PitShape = Literal["uniform", "u_shape", "hump", "skewed"]


def crps_ensemble(observations: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """
    Ensemble-form CRPS per observation.

    Parameters
    ----------
    observations:
        ``(n,)`` or broadcastable to ensemble leading dim.
    ensemble:
        ``(n, m)`` members or ``(m,)`` for a single obs.
    """
    obs = np.asarray(observations, dtype=np.float64)
    ens = np.asarray(ensemble, dtype=np.float64)
    if ens.ndim == 1:
        return np.asarray([ps.crps_ensemble(float(obs.ravel()[0]), ens)], dtype=np.float64)
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be (n, m) or (m,), got {ens.shape}")
    n = ens.shape[0]
    if obs.size == 1:
        obs = np.full(n, float(obs.ravel()[0]), dtype=np.float64)
    elif obs.shape[0] != n:
        raise ValueError(f"obs length {obs.shape[0]} != ensemble rows {n}")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = ens[i]
        row = row[np.isfinite(row)]
        if row.size == 0:
            out[i] = np.nan
        else:
            out[i] = ps.crps_ensemble(float(obs[i]), row)
    return out


def crps_quantile(
    observations: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: np.ndarray | list[float],
) -> np.ndarray:
    """
    CRPS from predicted quantiles via CDF quadrature (properscoring).

    Parameters
    ----------
    quantile_preds:
        ``(n, Q)`` predicted quantile values.
    quantile_levels:
        ``(Q,)`` nominal levels in (0, 1), strictly increasing.
    """
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    preds = np.asarray(quantile_preds, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    if preds.shape[1] != levels.size:
        raise ValueError("quantile_preds width must match quantile_levels")
    order = np.argsort(levels)
    levels = levels[order]
    preds = preds[:, order]
    n = preds.shape[0]
    out = np.empty(n, dtype=np.float64)
    tau_grid = np.linspace(float(levels[0]), float(levels[-1]), 51)
    for i in range(n):
        members = np.interp(tau_grid, levels, preds[i])
        out[i] = ps.crps_ensemble(float(obs[i]), members)
    return out


def pit_values_from_intervals(
    observations: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> np.ndarray:
    """PIT under uniform predictive CDF on [lower, upper]."""
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    lo = np.asarray(lowers, dtype=np.float64).reshape(-1)
    hi = np.asarray(uppers, dtype=np.float64).reshape(-1)
    width = np.maximum(hi - lo, 1e-6)
    return np.clip((y - lo) / width, 0.0, 1.0)


def pit_values_from_quantiles(
    observations: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: np.ndarray | list[float],
) -> np.ndarray:
    """PIT via interpolated inverse CDF from quantile fan."""
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    preds = np.asarray(quantile_preds, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    order = np.argsort(levels)
    levels = levels[order]
    preds = preds[:, order]
    out = np.empty(len(y), dtype=np.float64)
    for i in range(len(y)):
        out[i] = float(np.interp(y[i], preds[i], levels, left=0.0, right=1.0))
    return np.clip(out, 0.0, 1.0)


def _classify_pit_shape(counts: np.ndarray, n: int, n_bins: int) -> PitShape:
    if n == 0:
        return "uniform"
    expected = n / n_bins
    edge = (counts[0] + counts[-1]) / max(n, 1)
    center = counts[n_bins // 2] / max(n, 1)
    edge_exp = 2.0 / n_bins
    center_exp = 1.0 / n_bins
    if edge > 1.5 * edge_exp and center < 0.7 * center_exp:
        return "u_shape"
    if center > 1.8 * center_exp and edge < 0.8 * edge_exp:
        return "hump"
    chi2_p = float(stats.chisquare(counts, f_exp=np.full(n_bins, expected)).pvalue)
    if chi2_p < 0.01:
        return "skewed"
    return "uniform"


def pit_histogram(
    observations: np.ndarray,
    *,
    lowers: np.ndarray | None = None,
    uppers: np.ndarray | None = None,
    quantiles: np.ndarray | None = None,
    quantile_levels: np.ndarray | list[float] | None = None,
    ensemble: np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    PIT histogram and reliability-oriented diagnostics.

    Returns
    -------
    bin_edges, counts, reliability_data, diagnostics
    """
    if lowers is not None and uppers is not None:
        pit = pit_values_from_intervals(observations, lowers, uppers)
    elif quantiles is not None and quantile_levels is not None:
        pit = pit_values_from_quantiles(observations, quantiles, quantile_levels)
    elif ensemble is not None:
        ens = np.asarray(ensemble, dtype=np.float64)
        obs = np.asarray(observations, dtype=np.float64).reshape(-1)
        pit = np.empty(len(obs), dtype=np.float64)
        if ens.ndim == 1:
            ens = ens.reshape(1, -1)
        for i in range(len(obs)):
            members = np.sort(ens[i if ens.shape[0] == len(obs) else 0])
            pit[i] = float(np.mean(members <= obs[i]))
    else:
        raise ValueError("pit_histogram requires intervals, quantiles, or ensemble")

    pit = pit[np.isfinite(pit)]
    counts, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    n = int(counts.sum())
    expected = n / n_bins if n > 0 else 1.0
    if n > 0:
        chi2 = stats.chisquare(counts, f_exp=np.full(n_bins, expected))
        chi2_stat = float(chi2.statistic)
        chi2_p = float(chi2.pvalue)
    else:
        chi2_stat = float("nan")
        chi2_p = float("nan")

    nominal = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    empirical = counts / max(n, 1)
    reliability_data = {
        "nominal": nominal.tolist(),
        "empirical": empirical.tolist(),
        "bin_edges": bin_edges.tolist(),
    }
    diagnostics: dict[str, Any] = {
        "pit_chi2_stat": chi2_stat,
        "pit_chi2_p": chi2_p,
        "shape": _classify_pit_shape(counts, n, n_bins),
        "n": n,
    }
    return bin_edges, counts, reliability_data, diagnostics


def reliability_diagram(
    observations: np.ndarray,
    predicted_quantiles: np.ndarray,
    quantile_levels: np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """
    Reliability: nominal quantile levels vs empirical P(y <= q_hat).

    Returns nominal, empirical, ECE, extra dict with sharpness per level.
    """
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    preds = np.asarray(predicted_quantiles, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    order = np.argsort(levels)
    levels = levels[order]
    preds = preds[:, order]
    nominal = levels
    empirical = np.array([np.mean(y <= preds[:, j]) for j in range(len(levels))], dtype=np.float64)
    ece = float(np.mean(np.abs(nominal - empirical)))
    widths = []
    if preds.shape[1] >= 2:
        widths = (preds[:, -1] - preds[:, 0]).tolist()
    extra = {
        "sharpness_per_row": widths,
        "mean_sharpness": float(np.mean(preds[:, -1] - preds[:, 0])) if preds.shape[1] >= 2 else float("nan"),
    }
    return nominal, empirical, ece, extra


def sharpness(
    predicted_intervals: tuple[np.ndarray, np.ndarray] | np.ndarray,
) -> float | np.ndarray:
    """
    Mean interval width (lower = sharper).

    ``predicted_intervals`` may be ``(lowers, uppers)`` or ``(n, 2)`` array.
    """
    if isinstance(predicted_intervals, tuple):
        lo, hi = predicted_intervals
        widths = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    else:
        arr = np.asarray(predicted_intervals, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2:
            widths = arr[:, 1] - arr[:, 0]
        else:
            raise ValueError("predicted_intervals must be (lowers, uppers) or (n, 2)")
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return float("nan")
    if widths.size == 1:
        return float(widths[0])
    return float(np.mean(widths))


def energy_score(
    observations_multivar: np.ndarray,
    ensemble_multivar: np.ndarray,
) -> np.ndarray:
    """
    Multivariate Energy Score per row.

    observations: ``(n, d)``; ensemble: ``(n, m, d)``.
    """
    obs = np.asarray(observations_multivar, dtype=np.float64)
    ens = np.asarray(ensemble_multivar, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if ens.ndim == 2:
        ens = ens.reshape(1, ens.shape[0], ens.shape[1])
    n = obs.shape[0]
    out = np.empty(n, dtype=np.float64)
    es_fn = getattr(sr, "es_ensemble", None) or sr.energy_score
    for i in range(n):
        out[i] = float(es_fn(obs[i], ens[i]))
    return out


def crpss(crps_model: float, crps_baseline: float) -> float:
    """CRPS skill score: 1 - CRPS_model / CRPS_baseline."""
    if not np.isfinite(crps_baseline) or crps_baseline <= 0:
        return float("nan")
    return float(1.0 - crps_model / crps_baseline)


def aggregate_metrics(by_stratum: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pool numeric metrics across strata (unweighted mean)."""
    if not by_stratum:
        return {}
    keys: set[str] = set()
    for v in by_stratum.values():
        keys.update(k for k, val in v.items() if isinstance(val, (int, float)))
    pooled: dict[str, float] = {}
    for key in keys:
        vals = [float(v[key]) for v in by_stratum.values() if key in v and np.isfinite(v[key])]
        if vals:
            pooled[key] = float(np.mean(vals))
    return pooled


__all__ = [
    "PitShape",
    "aggregate_metrics",
    "crps_ensemble",
    "crps_quantile",
    "crpss",
    "energy_score",
    "pit_histogram",
    "pit_values_from_intervals",
    "pit_values_from_quantiles",
    "reliability_diagram",
    "sharpness",
]
