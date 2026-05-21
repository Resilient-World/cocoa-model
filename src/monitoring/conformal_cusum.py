"""
Conformal CUSUM change-point detector on the IID / exchangeability null.

Sanity-check parallel to WCTM; reference Vovk et al., PMLR 266 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConformalCUSUMState:
    """Serializable CUSUM state."""

    statistic: float = 0.0
    null_mean: float = 0.0
    n_updates: int = 0
    alarm_active: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "statistic": self.statistic,
            "null_mean": self.null_mean,
            "n_updates": self.n_updates,
            "alarm_active": self.alarm_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalCUSUMState:
        return cls(
            statistic=float(data.get("statistic", 0.0)),
            null_mean=float(data.get("null_mean", 0.0)),
            n_updates=int(data.get("n_updates", 0)),
            alarm_active=bool(data.get("alarm_active", False)),
        )


class ConformalCUSUM:
    """
    One-sided CUSUM on centered conformity scores.

    S_t = max(0, S_{t-1} + score_t - k); alarm when S_t > h.
    """

    def __init__(
        self,
        *,
        k: float = 0.0,
        h: float = 5.0,
        burn_in: int = 20,
        ema_decay: float = 0.05,
    ) -> None:
        self.k = float(k)
        self.h = float(h)
        self.burn_in = int(burn_in)
        self.ema_decay = float(ema_decay)
        self._state = ConformalCUSUMState()

    @property
    def state(self) -> ConformalCUSUMState:
        return self._state

    def statistic(self) -> float:
        return self._state.statistic

    def update(self, score: float) -> float:
        """Ingest score; return current CUSUM statistic."""
        s = float(score)
        st = self._state
        st.n_updates += 1
        if st.n_updates <= self.burn_in:
            st.null_mean = (1.0 - self.ema_decay) * st.null_mean + self.ema_decay * s
        centered = s - st.null_mean - self.k
        st.statistic = max(0.0, st.statistic + centered)
        st.alarm_active = st.statistic > self.h
        return st.statistic

    def detect(self) -> bool:
        return self._state.alarm_active

    def load_state(self, state: ConformalCUSUMState | dict) -> None:
        if isinstance(state, ConformalCUSUMState):
            self._state = state
        else:
            self._state = ConformalCUSUMState.from_dict(state)


__all__ = ["ConformalCUSUM", "ConformalCUSUMState"]
