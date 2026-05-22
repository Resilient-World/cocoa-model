"""
Conformal PID control for time-series prediction intervals.

Reference: Angelopoulos, Candes, Tibshirani (2023), NeurIPS —
Conformal PID Control for Time Series Prediction.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Deque

from models.conformal.online_conformal_base import adaptive_learning_rate, miscoverage_indicator


class ConformalPID:
    """
    PID-style online conformal threshold.

    Update::

        q_{t+1} = g'_t(X_t) + η_t·(err_t − α) + r_t(Σ(err_i − α))

    where ``η_t = η · (max − min)`` over the last ``window`` scores,
    and ``r_t = K_I · tanh(saturation_factor · cumulative_error)`` limits integrator windup.

    If ``g_prime`` is None, ``g'_t = 0`` (pure quantile tracking + integral feedback).
    """

    def __init__(
        self,
        alpha: float,
        eta: float = 0.01,
        *,
        window: int = 100,
        K_I: float = 1.0,
        saturation_factor: float = 0.1,
        q_init: float = 0.0,
        g_prime: Callable[[float], float] | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.eta_base = float(eta)
        self.window = int(window)
        self.K_I = float(K_I)
        self.saturation_factor = float(saturation_factor)
        self.g_prime = g_prime
        self.q: float = float(q_init)
        self.t: int = 0
        self._scores: Deque[float] = deque(maxlen=10_000)
        self._cum_err: float = 0.0

    @property
    def current_threshold(self) -> float:
        return self.q

    def reset(self, *, q_init: float | None = None) -> None:
        if q_init is not None:
            self.q = float(q_init)
        self.t = 0
        self._scores.clear()
        self._cum_err = 0.0

    def _integrator(self) -> float:
        return self.K_I * math.tanh(self.saturation_factor * self._cum_err)

    def update(self, score: float, covered: bool | None = None) -> float:
        err = 1.0 - float(covered) if covered is not None else miscoverage_indicator(score, self.q)
        self._scores.append(float(score))
        self._cum_err += err - self.alpha
        eta_t = adaptive_learning_rate(self._scores, self.eta_base, window=self.window)
        g_term = self.g_prime(float(score)) if self.g_prime is not None else 0.0
        r_term = self._integrator()
        self.q = g_term + self.q + eta_t * (err - self.alpha) + r_term
        self.t += 1
        return self.q


__all__ = ["ConformalPID"]
