"""
Adaptive Conformal Inference (ACI) and multi-horizon extension.

References
----------
- Gibbs & Candes (2021), NeurIPS: Adaptive Conformal Inference Under Distribution Shift.
- Hallberg Szabadváry (2024), PMLR 230: Multi-step ACI for time-series forecasting.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from models.conformal.online_conformal_base import miscoverage_indicator

DEFAULT_SCENARIO_HORIZONS: tuple[str, ...] = ("2030", "2050", "2080")


class AdaptiveConformalInference:
    """
    Online threshold update for conformal prediction under distribution shift.

    Update (Gibbs & Candes 2021)::

        q_{t+1} = q_t + η · (err_t − α),   err_t = 1{s_t > q_t}

    **Finite-sample bound (Eq. 1, Gibbs & Candes 2021):** for T steps,

    ``|Ĉ − (1 − α)| ≤ (ε₁ + η) / (η T)`` where ``Ĉ`` is empirical coverage,
    ``ε₁`` is an initialization slack (``eps_1``), and ``η`` is the learning rate.

    Long-term miscoverage is driven toward ``α`` when the score stream is
    stationary within each regime.
    """

    def __init__(
        self,
        alpha: float,
        eta: float = 0.005,
        q_init: float = 0.0,
        *,
        eps_1: float = 0.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if eta <= 0.0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.eps_1 = float(eps_1)
        self.q: float = float(q_init)
        self.t: int = 0
        self._scores: deque[float] = deque(maxlen=10_000)
        self._errs: deque[float] = deque(maxlen=10_000)
        self._covered: deque[bool] = deque(maxlen=10_000)

    @property
    def current_threshold(self) -> float:
        return self.q

    def reset(self, *, q_init: float | None = None) -> None:
        if q_init is not None:
            self.q = float(q_init)
        self.t = 0
        self._scores.clear()
        self._errs.clear()
        self._covered.clear()

    def update(self, score: float, covered: bool | None = None) -> float:
        """
        Observe conformity score ``score`` and return updated threshold ``q_{t+1}``.

        If ``covered`` is None, miscoverage is ``score > q_t`` (pre-update threshold).
        """
        err = 1.0 - float(covered) if covered is not None else miscoverage_indicator(score, self.q)
        self._scores.append(float(score))
        self._errs.append(err)
        self._covered.append(err == 0.0)
        self.q = self.q + self.eta * (err - self.alpha)
        self.t += 1
        return self.q

    def finite_sample_bound_rhs(self, T: int | None = None) -> float:
        """Right-hand side of Gibbs & Candes Eq. (1): ``(eps_1 + eta) / (eta * T)``."""
        steps = T if T is not None else max(self.t, 1)
        return (self.eps_1 + self.eta) / (self.eta * steps)

    def empirical_coverage(self) -> float:
        if not self._covered:
            return float("nan")
        return float(np.mean(list(self._covered)))

    def coverage_trajectory(self) -> np.ndarray:
        """Cumulative mean coverage after each update."""
        if not self._covered:
            return np.array([], dtype=np.float64)
        cov = np.array(self._covered, dtype=np.float64)
        return np.cumsum(cov) / np.arange(1, len(cov) + 1)


class MultiStepACI:
    """
    Horizon-stratified ACI for multi-step / scenario forecasts.

    Maintains independent :class:`AdaptiveConformalInference` instances per horizon
    (default ``2030``, ``2050``, ``2080`` for ``/simulate-scenario``).

    **Theory (Hallberg Szabadváry 2024):** each horizon inherits the finite-sample
    ACI guarantee at that horizon; the joint long-run miscoverage across horizons
    is bounded under their Eq. (10) when horizons are updated with the same ``α``
    schedule (see paper for the aggregated bound).
    """

    def __init__(
        self,
        alpha: np.ndarray,
        eta: np.ndarray,
        horizons: list[str],
    ) -> None:
        horizons = list(horizons)
        alpha_arr = np.asarray(alpha, dtype=np.float64).reshape(-1)
        eta_arr = np.asarray(eta, dtype=np.float64).reshape(-1)
        if len(horizons) != len(alpha_arr) or len(horizons) != len(eta_arr):
            raise ValueError("alpha, eta, and horizons must have the same length")
        self.horizons = horizons
        self._updaters: dict[str, AdaptiveConformalInference] = {
            h: AdaptiveConformalInference(float(a), eta=float(e))
            for h, a, e in zip(horizons, alpha_arr, eta_arr, strict=True)
        }

    def update(
        self,
        score_vec: np.ndarray,
        covered_vec: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Apply ACI per horizon; return new thresholds keyed by horizon.
        """
        scores = np.asarray(score_vec, dtype=np.float64).reshape(-1)
        if len(scores) != len(self.horizons):
            raise ValueError(f"score_vec length {len(scores)} != {len(self.horizons)} horizons")
        covered: np.ndarray | None = None
        if covered_vec is not None:
            covered = np.asarray(covered_vec, dtype=bool).reshape(-1)
            if len(covered) != len(self.horizons):
                raise ValueError("covered_vec length must match horizons")
        out: dict[str, float] = {}
        for i, h in enumerate(self.horizons):
            cov_i = None if covered is None else bool(covered[i])
            out[h] = self._updaters[h].update(float(scores[i]), covered=cov_i)
        return out

    def threshold(self, horizon: str) -> float:
        return self._updaters[horizon].current_threshold

    def empirical_coverage_by_horizon(self) -> dict[str, float]:
        return {h: u.empirical_coverage() for h, u in self._updaters.items()}


def default_multistep_aci(
    alpha: float = 0.1,
    eta: float = 0.005,
    horizons: list[str] | None = None,
) -> MultiStepACI:
    """Factory with uniform ``α`` and ``η`` across scenario horizons."""
    hz = horizons or list(DEFAULT_SCENARIO_HORIZONS)
    n = len(hz)
    return MultiStepACI(
        alpha=np.full(n, alpha),
        eta=np.full(n, eta),
        horizons=hz,
    )


__all__ = [
    "DEFAULT_SCENARIO_HORIZONS",
    "AdaptiveConformalInference",
    "MultiStepACI",
    "default_multistep_aci",
]
