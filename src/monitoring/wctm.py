"""
Weighted Conformal Test Martingales (WCTM) for drift monitoring.

Reference: Prinster, Han, Liu & Saria, "WATCH: Adaptive Monitoring for AI Deployments
via Weighted-Conformal Martingales", ICML 2025 (PMLR 267:49830-49859).

Betting function and composite-jumper martingale follow Section 2.3 / Eq. (6)-(7).
Root-cause rules follow Section 3.6 (parallel label-WCTM and X-CTM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Sequence

DriftDiagnosis = Literal["none", "covariate_shift", "concept_shift", "out_of_support"]

EPSILON_CHOICES = (-1, 0, 1)


def betting_function(p: float, epsilon: int) -> float:
    """Eq. (6): h_epsilon(p) = 1 + epsilon * (p - 0.5)."""
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    return max(1e-12, 1.0 + float(epsilon) * (p - 0.5))


def weighted_conformal_pvalue(
    score: float,
    calibration_scores: Sequence[float],
    calibration_weights: Sequence[float] | None = None,
    *,
    current_weight: float = 1.0,
) -> float:
    """
    Laplace-style weighted conformal p-value: fraction of calibration scores >= score.

    Includes the current observation in the denominator (conservative).
    """
    if not calibration_scores:
        return 1.0
    weights = list(calibration_weights or [1.0] * len(calibration_scores))
    if len(weights) != len(calibration_scores):
        weights = [1.0] * len(calibration_scores)
    num = sum(w for s, w in zip(calibration_scores, weights, strict=True) if s >= score)
    num += current_weight
    den = sum(weights) + current_weight
    return float(min(1.0, max(1e-12, num / den)))


def score_from_yield_observation(y_obs: float, y_pred: float, sigma_t: float) -> float:
    """Normalized nonconformity |y_obs - y_pred| / sigma_t."""
    sigma = max(float(sigma_t), 1e-6)
    return abs(float(y_obs) - float(y_pred)) / sigma


def sigma_from_interval(lo: float, hi: float, *, fallback: float = 1e-3) -> float:
    """Scale from half-interval width."""
    return max((float(hi) - float(lo)) / 2.0, fallback)


@dataclass
class DriftAlarm:
    """Active drift alarm."""

    diagnosis: DriftDiagnosis
    log_martingale: float
    triggered_at: str
    wealth_ratio: float = 1.0

    def to_payload(self) -> dict[str, str | float]:
        return {
            "type": self.diagnosis,
            "log_martingale": round(self.log_martingale, 4),
            "triggered_at": self.triggered_at,
        }


@dataclass
class CompositeJumperState:
    """Wealth per epsilon branch for composite jumper (Vovk et al. 2022)."""

    wealth_by_epsilon: dict[str, float] = field(
        default_factory=lambda: {"-1": 1.0, "0": 1.0, "1": 1.0}
    )
    last_epsilon: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "wealth_-1": self.wealth_by_epsilon.get("-1", 1.0),
            "wealth_0": self.wealth_by_epsilon.get("0", 1.0),
            "wealth_1": self.wealth_by_epsilon.get("1", 1.0),
            "last_epsilon": self.last_epsilon,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompositeJumperState:
        return cls(
            wealth_by_epsilon={
                "-1": float(data.get("wealth_-1", 1.0)),
                "0": float(data.get("wealth_0", 1.0)),
                "1": float(data.get("wealth_1", 1.0)),
            },
            last_epsilon=int(data.get("last_epsilon", 0)),
        )


class WeightedConformalTestMartingale:
    """
    Online WCTM with composite-jumper betting on weighted conformal p-values.

    Parameters
    ----------
    alpha_fpr:
        Target false-alarm rate; alarm when log-wealth exceeds log(1/alpha_fpr).
    score_cap:
        Scores above this trigger ``out_of_support`` diagnosis.
    """

    def __init__(
        self,
        *,
        alpha_fpr: float = 0.01,
        score_cap: float = 8.0,
        m0: float = 1.0,
    ) -> None:
        if not 0.0 < alpha_fpr < 1.0:
            raise ValueError(f"alpha_fpr must be in (0, 1), got {alpha_fpr}")
        self.alpha_fpr = float(alpha_fpr)
        self.score_cap = float(score_cap)
        self.m0 = float(m0)
        self.log_threshold = math.log(1.0 / self.alpha_fpr)
        self._log_martingale = 0.0
        self._composite = CompositeJumperState()
        self._calibration_scores: list[float] = []
        self._calibration_weights: list[float] = []
        self._last_score: float | None = None
        self._last_p: float | None = None
        self._active_alarm: DriftAlarm | None = None
        self._x_alarm_active: bool = False

    @property
    def log_martingale(self) -> float:
        return self._log_martingale

    @property
    def wealth_ratio(self) -> float:
        return math.exp(self._log_martingale)

    @property
    def composite_state(self) -> CompositeJumperState:
        return self._composite

    def set_calibration_window(
        self,
        scores: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> None:
        self._calibration_scores = [float(s) for s in scores]
        self._calibration_weights = (
            [float(w) for w in weights]
            if weights is not None
            else [1.0] * len(self._calibration_scores)
        )

    def append_calibration(self, score: float, weight: float = 1.0) -> None:
        self._calibration_scores.append(float(score))
        self._calibration_weights.append(float(weight))

    def set_x_alarm_active(self, active: bool) -> None:
        """Set parallel X-CTM alarm state for :meth:`diagnose`."""
        self._x_alarm_active = bool(active)

    def _composite_wealth(self, p: float) -> float:
        """Eq. (7): integrate wealth over epsilon branches with simple jumper."""
        total = 0.0
        for eps in EPSILON_CHOICES:
            key = str(eps)
            w_prev = self._composite.wealth_by_epsilon.get(key, 1.0)
            h = betting_function(p, eps)
            self._composite.wealth_by_epsilon[key] = w_prev * h
            total += self._composite.wealth_by_epsilon[key]
        return total / len(EPSILON_CHOICES)

    def update(self, nonconformity_score: float, weight: float = 1.0) -> float:
        """
        Ingest a normalized nonconformity score; return log-martingale.

        Converts score to weighted conformal p-value, then applies betting update.
        """
        score = float(nonconformity_score)
        self._last_score = score
        p = weighted_conformal_pvalue(
            score,
            self._calibration_scores,
            self._calibration_weights,
            current_weight=weight,
        )
        self._last_p = p
        self.append_calibration(score, weight)

        wealth = self._composite_wealth(p)
        self._log_martingale = math.log(max(wealth, 1e-300) / self.m0)

        alarm = self.detect()
        if alarm is not None:
            self._active_alarm = alarm

        return self._log_martingale

    def detect(self) -> DriftAlarm | None:
        """Return alarm if log-martingale exceeds log(1/alpha_fpr) or out-of-support."""
        if self._last_score is not None and self._last_score > self.score_cap:
            return DriftAlarm(
                diagnosis="out_of_support",
                log_martingale=self._log_martingale,
                triggered_at=_utc_now_iso(),
                wealth_ratio=self.wealth_ratio,
            )
        if self._log_martingale > self.log_threshold:
            diagnosis = self.diagnose()
            if diagnosis == "none":
                diagnosis = "concept_shift"
            return DriftAlarm(
                diagnosis=diagnosis,
                log_martingale=self._log_martingale,
                triggered_at=_utc_now_iso(),
                wealth_ratio=self.wealth_ratio,
            )
        return None

    def diagnose(self) -> DriftDiagnosis:
        """
        Root-cause analysis (WATCH Sec. 3.6).

        Requires :meth:`set_x_alarm_active` from parallel X-monitor.
        """
        if self._last_score is not None and self._last_score > self.score_cap:
            return "out_of_support"
        if self._log_martingale <= self.log_threshold:
            return "none"
        if self._x_alarm_active:
            return "covariate_shift"
        return "concept_shift"

    @property
    def active_alarm(self) -> DriftAlarm | None:
        return self._active_alarm

    def load_state(
        self,
        *,
        log_martingale: float,
        composite: CompositeJumperState | dict | None = None,
        calibration_scores: Sequence[float] | None = None,
        calibration_weights: Sequence[float] | None = None,
    ) -> None:
        self._log_martingale = float(log_martingale)
        if composite is not None:
            if isinstance(composite, CompositeJumperState):
                self._composite = composite
            else:
                self._composite = CompositeJumperState.from_dict(composite)
        if calibration_scores is not None:
            self.set_calibration_window(calibration_scores, calibration_weights)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def covariate_nonconformity(
    feature_vector: Sequence[float],
    ema: Sequence[float] | None,
    *,
    ema_decay: float = 0.05,
) -> tuple[float, list[float]]:
    """
    L2 distance from rolling EMA of features; returns (score, updated_ema).

    Used by the parallel X-CTM path.
    """
    vec = [float(x) for x in feature_vector]
    if ema is None or len(ema) != len(vec):
        return 0.0, vec
    ema_f = [float(x) for x in ema]
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, ema_f, strict=True)))
    new_ema = [
        (1.0 - ema_decay) * e + ema_decay * v for e, v in zip(ema_f, vec, strict=True)
    ]
    return dist, new_ema


__all__ = [
    "CompositeJumperState",
    "DriftAlarm",
    "DriftDiagnosis",
    "WeightedConformalTestMartingale",
    "betting_function",
    "covariate_nonconformity",
    "score_from_yield_observation",
    "sigma_from_interval",
    "weighted_conformal_pvalue",
]
