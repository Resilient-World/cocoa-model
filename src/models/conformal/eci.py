"""
Error-quantified Conformal Inference (ECI) and variants.

Reference: Wu, Hu, Bao, Xia, Zou (2025), ICLR —
Error-quantified Conformal Inference for Time Series.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from models.conformal.online_conformal_base import (
    adaptive_learning_rate,
    miscoverage_indicator,
    sigmoid_derivative,
)


class ErrorQuantifiedConformalInference:
    """
    ECI (Algorithm 1, Wu et al. 2025).

    Update::

        q_{t+1} = q_t + η_t · [err_t − α + (s_t − q_t) · f'(s_t − q_t)]

    with ``f(x) = σ(c·x)`` (sigmoid), default ``c = 1``.

    Assumes bounded scores ``s_t ∈ [0, B]`` (Assumption 1 in paper).
    """

    def __init__(
        self,
        alpha: float,
        eta: float = 0.01,
        *,
        c: float = 1.0,
        window: int = 100,
        q_init: float = 0.0,
        adaptive_eta: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.eta_base = float(eta)
        self.c = float(c)
        self.window = int(window)
        self.adaptive_eta = adaptive_eta
        self.q: float = float(q_init)
        self.t: int = 0
        self._scores: Deque[float] = deque(maxlen=10_000)

    @property
    def current_threshold(self) -> float:
        return self.q

    def reset(self, *, q_init: float | None = None) -> None:
        if q_init is not None:
            self.q = float(q_init)
        self.t = 0
        self._scores.clear()

    def _eq_feedback(self, score: float, q: float) -> float:
        gap = score - q
        return gap * float(sigmoid_derivative(gap, c=self.c))

    def update(self, score: float, covered: bool | None = None) -> float:
        err = 1.0 - float(covered) if covered is not None else miscoverage_indicator(score, self.q)
        self._scores.append(float(score))
        eta_t = (
            adaptive_learning_rate(self._scores, self.eta_base, window=self.window)
            if self.adaptive_eta
            else self.eta_base
        )
        feedback = (err - self.alpha) + self._eq_feedback(score, self.q)
        self.q = self.q + eta_t * feedback
        self.t += 1
        return self.q


class ECICutoff(ErrorQuantifiedConformalInference):
    """
    ECI with cutoff (Algorithm 2): EQ term active only when ``|s_t − q_t| > h_t``,

    ``h_t = h · (max_{window} s − min_{window} s)``.
    """

    def __init__(
        self,
        alpha: float,
        eta: float = 0.01,
        *,
        c: float = 1.0,
        h: float = 0.5,
        window: int = 100,
        q_init: float = 0.0,
    ) -> None:
        super().__init__(alpha, eta, c=c, window=window, q_init=q_init)
        self.h = float(h)

    def _cutoff_scale(self) -> float:
        if not self._scores:
            return 0.0
        recent = list(self._scores)[-self.window :]
        return self.h * (max(recent) - min(recent))

    def update(self, score: float, covered: bool | None = None) -> float:
        err = 1.0 - float(covered) if covered is not None else miscoverage_indicator(score, self.q)
        self._scores.append(float(score))
        eta_t = (
            adaptive_learning_rate(self._scores, self.eta_base, window=self.window)
            if self.adaptive_eta
            else self.eta_base
        )
        eq = self._eq_feedback(score, self.q)
        if abs(score - self.q) <= self._cutoff_scale():
            eq = 0.0
        self.q = self.q + eta_t * ((err - self.alpha) + eq)
        self.t += 1
        return self.q


class ECIIntegral(ErrorQuantifiedConformalInference):
    """
    ECI with historical integration (Algorithm 3).

    ``q_{t+1} = q_t + η_t · Σ_i w_i · [err_i − α + (s_i − q_i)·f'(s_i − q_i)]``

    with ``w_i ∝ 0.95^{t-i}`` normalized to sum 1.
    """

    def __init__(
        self,
        alpha: float,
        eta: float = 0.01,
        *,
        c: float = 1.0,
        decay: float = 0.95,
        window: int = 100,
        q_init: float = 0.0,
    ) -> None:
        super().__init__(alpha, eta, c=c, window=window, q_init=q_init)
        self.decay = float(decay)
        self._history_scores: Deque[float] = deque(maxlen=2_000)
        self._history_qs: Deque[float] = deque(maxlen=2_000)
        self._history_err: Deque[float] = deque(maxlen=2_000)

    def reset(self, *, q_init: float | None = None) -> None:
        super().reset(q_init=q_init)
        self._history_scores.clear()
        self._history_qs.clear()
        self._history_err.clear()

    def update(self, score: float, covered: bool | None = None) -> float:
        err = 1.0 - float(covered) if covered is not None else miscoverage_indicator(score, self.q)
        q_prev = self.q
        self._scores.append(float(score))
        self._history_scores.append(float(score))
        self._history_qs.append(q_prev)
        self._history_err.append(err)

        n = len(self._history_scores)
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = self.decay**ages
        weights /= weights.sum()

        integrated = 0.0
        for w, s_i, q_i, e_i in zip(
            weights,
            self._history_scores,
            self._history_qs,
            self._history_err,
            strict=True,
        ):
            integrated += w * ((e_i - self.alpha) + (s_i - q_i) * float(sigmoid_derivative(s_i - q_i, c=self.c)))

        eta_t = (
            adaptive_learning_rate(self._scores, self.eta_base, window=self.window)
            if self.adaptive_eta
            else self.eta_base
        )
        self.q = self.q + eta_t * integrated
        self.t += 1
        return self.q


__all__ = [
    "ECICutoff",
    "ECIIntegral",
    "ErrorQuantifiedConformalInference",
]
