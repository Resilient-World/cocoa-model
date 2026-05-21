"""Shared helpers for online conformal tests."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from models.aci import AdaptiveConformalInference
from models.conformal_pid import ConformalPID
from models.cqr import ConformalCalibrator
from models.online_conformal_base import conformal_quantile
from models.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from models.online_conformal_base import empirical_coverage, interval_from_q


class OnlineUpdater(Protocol):
    def reset(self, *, q_init: float | None = None) -> None: ...
    def update(self, score: float, covered: bool | None = None) -> float: ...
    @property
    def current_threshold(self) -> float: ...


def run_online_coverage(
    updater: OnlineUpdater,
    scores: np.ndarray,
    *,
    alpha: float,
    q_lo: float = -1.0,
    q_hi: float = 1.0,
    burn_in: int = 50,
    warm_start: int = 0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream scores through updater; return score-space coverage, lowers, uppers, thresholds.

    Coverage is ``mean(s_t <= q_t)`` using the pre-update threshold at each step
    (standard online conformal miscoverage protocol).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(scores)
    lowers = np.empty(n)
    uppers = np.empty(n)
    qs = np.empty(n)
    covered = np.empty(n, dtype=bool)
    q_init = 0.0
    if warm_start > 0:
        q_init = conformal_quantile(scores[:warm_start], alpha)
    updater.reset(q_init=q_init)
    for t in range(warm_start):
        q = updater.current_threshold
        qs[t] = q
        covered[t] = scores[t] <= q
        lo, _, hi = interval_from_q(q, q_lo, q_hi)
        lowers[t] = lo
        uppers[t] = hi
        updater.update(float(scores[t]))
    for t in range(warm_start, n):
        q = updater.current_threshold
        qs[t] = q
        covered[t] = scores[t] <= q
        lo, _, hi = interval_from_q(q, q_lo, q_hi)
        lowers[t] = lo
        uppers[t] = hi
        updater.update(float(scores[t]))
    eval_slice = slice(burn_in, n)
    cov = float(np.mean(covered[eval_slice]))
    return cov, lowers, uppers, qs


def distribution_shift_scores(
    T: int = 1000,
    shift_at: int = 500,
    *,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0.0, 1.0, shift_at)
    s2 = rng.normal(2.0, 1.0, T - shift_at)
    return np.concatenate([s1, s2]).astype(np.float64)


def split_cqr_static_coverage(
    scores: np.ndarray,
    *,
    alpha: float,
    cal_fraction: float = 0.5,
) -> float:
    """Calibrate Q on prefix only; evaluate score-space coverage on suffix (fails under shift)."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    n_cal = int(n * cal_fraction)
    cal = scores[:n_cal]
    test = scores[n_cal:]
    Q = ConformalCalibrator._conformal_quantile(cal, alpha)
    return float(np.mean(test <= Q))


def post_shift_coverage(
    scores: np.ndarray,
    qs: np.ndarray,
    *,
    shift_at: int,
    window: int = 100,
) -> float:
    """Coverage on the last ``window`` steps after ``shift_at``."""
    scores = np.asarray(scores)
    qs = np.asarray(qs)
    sl = slice(max(shift_at, len(scores) - window), len(scores))
    return float(np.mean(scores[sl] <= qs[sl]))


def synthetic_prophet_like_scores(n: int, *, seed: int, vol: float = 0.4) -> np.ndarray:
    """Exponential scores with mild AR persistence (finance-like residuals)."""
    rng = np.random.default_rng(seed)
    base = rng.exponential(scale=1.0 / max(vol, 0.1), size=n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.05)
    return np.maximum(base + ar, 0.0).astype(np.float64)
