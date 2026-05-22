"""
Shared helpers for online conformal threshold updates (ACI, PID, ECI).

Pure NumPy; no external conformal libraries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from models.conformal.cqr import ConformalCalibrator


def sigmoid(x: np.ndarray | float, *, c: float = 1.0) -> np.ndarray:
    """σ(cx) with numerical stability."""
    z = np.asarray(x, dtype=np.float64) * c
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(x: np.ndarray | float, *, c: float = 1.0) -> np.ndarray:
    """d/dx σ(cx) = c·σ(cx)·(1 - σ(cx))."""
    s = sigmoid(x, c=c)
    return c * s * (1.0 - s)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile; delegates to :class:`ConformalCalibrator`."""
    return ConformalCalibrator._conformal_quantile(scores, alpha)


def miscoverage_indicator(score: float, q: float) -> float:
    """``err = 1{s > q}`` as float."""
    return 1.0 if score > q else 0.0


def adaptive_learning_rate(
    scores_window: deque[float] | list[float],
    eta_base: float,
    *,
    window: int = 100,
) -> float:
    """
    Scale learning rate by recent score range (Wu et al. 2025 / Angelopoulos PID).

    ``eta_t = eta_base * (max(scores) - min(scores))`` over the last ``window`` scores.
    """
    if not scores_window:
        return float(eta_base)
    recent = list(scores_window)[-window:]
    span = max(recent) - min(recent)
    return float(eta_base * max(span, 1e-8))


def interval_from_q(q: float, q_lo: float, q_hi: float) -> tuple[float, float, float]:
    """Symmetric CQR inflation: ``[q_lo - q, q_med implied, q_hi + q]``."""
    return q_lo - q, (q_lo + q_hi) / 2.0, q_hi + q


def empirical_coverage(
    y: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> float:
    """Fraction of points with ``lower <= y <= upper``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(lowers, dtype=np.float64).reshape(-1)
    hi = np.asarray(uppers, dtype=np.float64).reshape(-1)
    return float(np.mean((y >= lo) & (y <= hi)))


def rolling_coverage(
    covered: np.ndarray,
    *,
    window: int = 50,
) -> np.ndarray:
    """Rolling mean of binary coverage indicators."""
    c = np.asarray(covered, dtype=np.float64).reshape(-1)
    if len(c) == 0:
        return c
    out = np.empty_like(c)
    for t in range(len(c)):
        start = max(0, t - window + 1)
        out[t] = c[start : t + 1].mean()
    return out


def pit_values(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Probability integral transform values ``F̂(s_t) ≈ 1{score <= q_t}`` for PIT diagnostics.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    q = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    return (s <= q).astype(np.float64)


def mean_interval_width(lowers: np.ndarray, uppers: np.ndarray) -> float:
    """Average ``upper - lower``."""
    lo = np.asarray(lowers, dtype=np.float64).reshape(-1)
    hi = np.asarray(uppers, dtype=np.float64).reshape(-1)
    return float(np.mean(hi - lo))


__all__ = [
    "adaptive_learning_rate",
    "conformal_quantile",
    "empirical_coverage",
    "interval_from_q",
    "mean_interval_width",
    "miscoverage_indicator",
    "pit_values",
    "rolling_coverage",
    "sigmoid",
    "sigmoid_derivative",
]
