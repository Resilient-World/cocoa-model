"""
Persist WCTM / CUSUM drift state per conformal stratum.

Mirrors :mod:`api.online_conformal_store` (Redis ``drift_monitoring_state`` or JSON file).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from api.online_conformal_store import stratum_key
from monitoring.conformal_cusum import ConformalCUSUM
from monitoring.wctm import WeightedConformalTestMartingale

log = structlog.get_logger(__name__)

SCORE_WINDOW_MAX = 500
FEATURE_EMA_MAX = 128


@dataclass
class DriftStratumState:
    log_martingale: float = 0.0
    x_log_martingale: float = 0.0
    composite_jumper: dict[str, float | int] = field(default_factory=dict)
    x_composite_jumper: dict[str, float | int] = field(default_factory=dict)
    calibration_scores: list[float] = field(default_factory=list)
    calibration_weights: list[float] = field(default_factory=list)
    x_calibration_scores: list[float] = field(default_factory=list)
    feature_ema: list[float] = field(default_factory=list)
    cusum: dict[str, float | int | bool] = field(default_factory=dict)
    last_alarm_at: str | None = None
    last_diagnosis: str = "none"
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_martingale": self.log_martingale,
            "x_log_martingale": self.x_log_martingale,
            "composite_jumper": self.composite_jumper,
            "x_composite_jumper": self.x_composite_jumper,
            "calibration_scores": self.calibration_scores[-SCORE_WINDOW_MAX:],
            "calibration_weights": self.calibration_weights[-SCORE_WINDOW_MAX:],
            "x_calibration_scores": self.x_calibration_scores[-SCORE_WINDOW_MAX:],
            "feature_ema": self.feature_ema[:FEATURE_EMA_MAX],
            "cusum": self.cusum,
            "last_alarm_at": self.last_alarm_at,
            "last_diagnosis": self.last_diagnosis,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftStratumState:
        return cls(
            log_martingale=float(data.get("log_martingale", 0.0)),
            x_log_martingale=float(data.get("x_log_martingale", 0.0)),
            composite_jumper=dict(data.get("composite_jumper") or {}),
            x_composite_jumper=dict(data.get("x_composite_jumper") or {}),
            calibration_scores=list(data.get("calibration_scores") or []),
            calibration_weights=list(data.get("calibration_weights") or []),
            x_calibration_scores=list(data.get("x_calibration_scores") or []),
            feature_ema=list(data.get("feature_ema") or []),
            cusum=dict(data.get("cusum") or {}),
            last_alarm_at=data.get("last_alarm_at"),
            last_diagnosis=str(data.get("last_diagnosis", "none")),
            update_count=int(data.get("update_count", 0)),
        )


class DriftStore:
    """Load/save per-stratum WCTM and CUSUM detectors."""

    def __init__(
        self,
        *,
        state_path: Path,
        redis_url: str | None = None,
        alpha_fpr: float = 0.01,
        score_cap: float = 8.0,
        cusum_h: float = 5.0,
        cusum_k: float = 0.0,
    ) -> None:
        self.state_path = Path(state_path)
        self.redis_url = redis_url
        self.alpha_fpr = alpha_fpr
        self.score_cap = score_cap
        self.cusum_h = cusum_h
        self.cusum_k = cusum_k
        self._stratum_cache: dict[str, DriftStratumState] = {}
        self._wctm_cache: dict[str, WeightedConformalTestMartingale] = {}
        self._x_wctm_cache: dict[str, WeightedConformalTestMartingale] = {}
        self._cusum_cache: dict[str, ConformalCUSUM] = {}
        self._redis_client: Any = None
        if redis_url:
            try:
                import redis

                self._redis_client = redis.from_url(redis_url, decode_responses=True)
            except ImportError:
                log.warning("redis not installed; drift store uses JSON only")

    def _read_all_json(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        with self.state_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _write_all_json(self, data: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.state_path)

    def _load_blob(self) -> dict[str, Any]:
        if self._redis_client is not None:
            raw = self._redis_client.get("drift_monitoring_state")
            if not raw:
                return {}
            return json.loads(raw)
        return self._read_all_json()

    def _save_blob(self, data: dict[str, Any]) -> None:
        if self._redis_client is not None:
            self._redis_client.set("drift_monitoring_state", json.dumps(data))
        else:
            self._write_all_json(data)

    def get_stratum_state(self, key: str) -> DriftStratumState:
        if key in self._stratum_cache:
            return self._stratum_cache[key]
        blob = self._load_blob()
        raw = blob.get(key)
        state = DriftStratumState.from_dict(raw) if raw else DriftStratumState()
        self._stratum_cache[key] = state
        return state

    def _hydrate_wctm(self, key: str, *, x: bool = False) -> WeightedConformalTestMartingale:
        cache = self._x_wctm_cache if x else self._wctm_cache
        if key in cache:
            return cache[key]
        st = self.get_stratum_state(key)
        w = WeightedConformalTestMartingale(
            alpha_fpr=self.alpha_fpr,
            score_cap=self.score_cap,
        )
        if x:
            w.load_state(
                log_martingale=st.x_log_martingale,
                composite=st.x_composite_jumper or None,
                calibration_scores=st.x_calibration_scores,
            )
        else:
            w.load_state(
                log_martingale=st.log_martingale,
                composite=st.composite_jumper or None,
                calibration_scores=st.calibration_scores,
                calibration_weights=st.calibration_weights,
            )
        cache[key] = w
        return w

    def get_wctm(self, key: str) -> WeightedConformalTestMartingale:
        return self._hydrate_wctm(key, x=False)

    def get_x_wctm(self, key: str) -> WeightedConformalTestMartingale:
        return self._hydrate_wctm(key, x=True)

    def get_cusum(self, key: str) -> ConformalCUSUM:
        if key in self._cusum_cache:
            return self._cusum_cache[key]
        st = self.get_stratum_state(key)
        c = ConformalCUSUM(k=self.cusum_k, h=self.cusum_h)
        if st.cusum:
            c.load_state(st.cusum)
        self._cusum_cache[key] = c
        return c

    def save_after_update(
        self,
        key: str,
        *,
        wctm: WeightedConformalTestMartingale,
        x_wctm: WeightedConformalTestMartingale,
        cusum: ConformalCUSUM,
        diagnosis: str,
        alarm_at: str | None,
        feature_ema: list[float] | None = None,
    ) -> DriftStratumState:
        st = self.get_stratum_state(key)
        st.log_martingale = wctm.log_martingale
        st.x_log_martingale = x_wctm.log_martingale
        st.composite_jumper = wctm.composite_state.to_dict()
        st.x_composite_jumper = x_wctm.composite_state.to_dict()
        st.calibration_scores = list(wctm._calibration_scores)[-SCORE_WINDOW_MAX:]
        st.calibration_weights = list(wctm._calibration_weights)[-SCORE_WINDOW_MAX:]
        st.x_calibration_scores = list(x_wctm._calibration_scores)[-SCORE_WINDOW_MAX:]
        st.cusum = cusum.state.to_dict()
        st.last_diagnosis = diagnosis
        st.last_alarm_at = alarm_at
        st.update_count += 1
        if feature_ema is not None:
            st.feature_ema = feature_ema[:FEATURE_EMA_MAX]
        self._stratum_cache[key] = st
        self._wctm_cache[key] = wctm
        self._x_wctm_cache[key] = x_wctm
        self._cusum_cache[key] = cusum

        blob = self._load_blob()
        blob[key] = st.to_dict()
        self._save_blob(blob)
        return st

    def get_drift_status(
        self,
        key: str,
        *,
        coverage_running_avg: float | None = None,
    ) -> dict[str, Any]:
        """Dashboard payload dict (mapped to :class:`api.schemas.DriftStatus`)."""
        w = self.get_wctm(key)
        st = self.get_stratum_state(key)
        cusum = self.get_cusum(key)
        alarm_active = (
            w.log_martingale > w.log_threshold
            or st.last_diagnosis not in ("none", "")
            or cusum.detect()
        )
        return {
            "stratum_key": key,
            "log_martingale": float(w.log_martingale),
            "alarm_active": bool(alarm_active),
            "diagnosis": st.last_diagnosis if alarm_active else "none",
            "coverage_running_avg": coverage_running_avg,
            "cusum_active": cusum.detect(),
        }

    def reload_from_disk(self) -> None:
        self._stratum_cache.clear()
        self._wctm_cache.clear()
        self._x_wctm_cache.clear()
        self._cusum_cache.clear()


def build_drift_store_from_settings(settings: Any) -> DriftStore:
    return DriftStore(
        state_path=Path(
            getattr(settings, "drift_state_path", "data/processed/drift_monitoring_state.json")
        ),
        redis_url=getattr(settings, "redis_url", None),
        alpha_fpr=float(getattr(settings, "drift_alpha_fpr", 0.01)),
        score_cap=float(getattr(settings, "drift_score_cap", 8.0)),
        cusum_h=float(getattr(settings, "drift_cusum_h", 5.0)),
        cusum_k=float(getattr(settings, "drift_cusum_k", 0.0)),
    )


__all__ = [
    "SCORE_WINDOW_MAX",
    "DriftStore",
    "DriftStratumState",
    "build_drift_store_from_settings",
    "stratum_key",
]
