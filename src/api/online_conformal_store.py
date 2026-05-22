"""
Persist online conformal thresholds and rolling coverage for /simulate-scenario strata.

Stratum key: ``{scenario}:{horizon_year}:{region}`` (48 FDP × SSP × horizon combinations).
CorrDiff traffic uses suffix ``:corrdiff`` on the same triple.
Uses Redis when ``REDIS_URL`` is set; otherwise atomic JSON at ``online_conformal_state_path``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from models.aci import AdaptiveConformalInference
from models.conformal_pid import ConformalPID
from models.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from models.quantile_yield_surrogate_online import OnlineMethod, _build_updater

log = structlog.get_logger(__name__)

ConformalMethod = Literal[
    "split_cqr",
    "aci",
    "conformal_pid",
    "eci",
    "eci_integral",
]

_UPDATER_TYPES = (
    AdaptiveConformalInference,
    ConformalPID,
    ErrorQuantifiedConformalInference,
    ECICutoff,
    ECIIntegral,
)

COVERAGE_WINDOW_MAX = 1000


def stratum_key(
    scenario: str,
    horizon_year: int | str,
    region: str,
    *,
    downscaling_method: str = "linear_delta",
) -> str:
    base = f"{scenario}:{int(horizon_year)}:{region}"
    return f"{base}:corrdiff" if downscaling_method == "corrdiff" else base


@dataclass
class StratumState:
    q_t: float = 0.0
    method: ConformalMethod = "eci_integral"
    coverage_window: list[bool] = field(default_factory=list)
    update_count: int = 0

    def coverage_running_avg(self) -> float | None:
        if not self.coverage_window:
            return None
        return float(sum(self.coverage_window) / len(self.coverage_window))

    def append_coverage(self, covered: bool) -> None:
        window = self.coverage_window
        if len(window) >= COVERAGE_WINDOW_MAX:
            window.pop(0)
        window.append(bool(covered))
        self.update_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_t": self.q_t,
            "method": self.method,
            "coverage_window": self.coverage_window,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StratumState:
        return cls(
            q_t=float(data.get("q_t", 0.0)),
            method=data.get("method", "eci_integral"),
            coverage_window=list(data.get("coverage_window") or []),
            update_count=int(data.get("update_count", 0)),
        )


class OnlineConformalStore:
    """Load/save per-stratum ``q_t`` and rolling coverage; in-memory updater cache."""

    def __init__(
        self,
        *,
        state_path: Path,
        initial_state_path: Path | None = None,
        redis_url: str | None = None,
        conformal_method: ConformalMethod = "eci_integral",
        alpha: float = 0.1,
        eci_eta: float = 2.5,
        eci_decay: float = 0.95,
        eci_window: int = 100,
        aci_eta: float = 0.005,
        pid_eta: float = 0.01,
    ) -> None:
        self.state_path = Path(state_path)
        self.initial_state_path = Path(initial_state_path) if initial_state_path else None
        self.redis_url = redis_url
        self.conformal_method = conformal_method
        self.alpha = alpha
        self.eci_eta = eci_eta
        self.eci_decay = eci_decay
        self.eci_window = eci_window
        self.aci_eta = aci_eta
        self.pid_eta = pid_eta
        self._initial_cache: dict[str, dict[str, Any]] | None = None
        self._updater_cache: dict[str, _UPDATER_TYPES] = {}
        self._stratum_cache: dict[str, StratumState] = {}
        self._redis_client: Any = None
        if redis_url:
            try:
                import redis

                self._redis_client = redis.from_url(redis_url, decode_responses=True)
            except ImportError:
                log.warning("redis package not installed; falling back to JSON state file")

    def _load_initial_blob(self) -> dict[str, dict[str, Any]]:
        if self._initial_cache is not None:
            return self._initial_cache
        path = self.initial_state_path
        if path is None or not path.is_file():
            self._initial_cache = {}
            return self._initial_cache
        with path.open(encoding="utf-8") as f:
            blob = json.load(f)
        self._initial_cache = blob if isinstance(blob, dict) else {}
        return self._initial_cache

    def q_init_for_key(self, key: str) -> float:
        blob = self._load_initial_blob()
        entry = blob.get(key) or {}
        return float(entry.get("q_t", entry.get("q_init", 0.0)))

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

    def _redis_get_blob(self) -> dict[str, Any]:
        if self._redis_client is None:
            return self._read_all_json()
        raw = self._redis_client.get("online_conformal_state")
        if not raw:
            return {}
        return json.loads(raw)

    def _redis_set_blob(self, data: dict[str, Any]) -> None:
        if self._redis_client is None:
            self._write_all_json(data)
            return
        self._redis_client.set("online_conformal_state", json.dumps(data))

    def _load_blob(self) -> dict[str, Any]:
        if self._redis_client is not None:
            return self._redis_get_blob()
        return self._read_all_json()

    def get_stratum_state(self, key: str) -> StratumState:
        if key in self._stratum_cache:
            return self._stratum_cache[key]
        blob = self._load_blob()
        raw = blob.get(key)
        if raw:
            state = StratumState.from_dict(raw)
        else:
            state = StratumState(
                q_t=self.q_init_for_key(key),
                method=self.conformal_method,
            )
        self._stratum_cache[key] = state
        return state

    def _method_kwargs(self, method: ConformalMethod) -> dict[str, Any]:
        if method in ("eci", "eci_integral", "eci_cutoff"):
            return {
                "eta": self.eci_eta,
                "decay": self.eci_decay,
                "window": self.eci_window,
            }
        if method == "aci":
            return {"eta": self.aci_eta}
        if method == "conformal_pid":
            return {"eta": self.pid_eta, "window": self.eci_window}
        return {}

    def get_updater(self, key: str, method: ConformalMethod | None = None) -> _UPDATER_TYPES:
        method = method or self.conformal_method
        cache_key = f"{key}:{method}"
        if cache_key in self._updater_cache:
            return self._updater_cache[cache_key]
        state = self.get_stratum_state(key)
        online_method: OnlineMethod = method  # type: ignore[assignment]
        if method == "split_cqr":
            online_method = "eci_integral"
        updater = _build_updater(
            online_method,
            self.alpha,
            q_init=state.q_t,
            **self._method_kwargs(method),
        )
        self._updater_cache[cache_key] = updater
        return updater

    def save_after_update(
        self,
        key: str,
        updater: _UPDATER_TYPES,
        *,
        covered: bool,
        method: ConformalMethod | None = None,
    ) -> StratumState:
        method = method or self.conformal_method
        state = self.get_stratum_state(key)
        state.q_t = float(updater.current_threshold)
        state.method = method
        state.append_coverage(covered)
        self._stratum_cache[key] = state

        blob = self._load_blob()
        blob[key] = state.to_dict()
        self._redis_set_blob(blob)
        cache_key = f"{key}:{method}"
        self._updater_cache[cache_key] = updater
        return state

    def coverage_running_avg(self, key: str) -> float | None:
        return self.get_stratum_state(key).coverage_running_avg()

    def reload_from_disk(self) -> None:
        """Drop in-memory caches (e.g. after API restart in tests)."""
        self._stratum_cache.clear()
        self._updater_cache.clear()


def build_store_from_settings(settings: Any) -> OnlineConformalStore:
    return OnlineConformalStore(
        state_path=settings.online_conformal_state_path,
        initial_state_path=settings.conformal_initial_state_path,
        redis_url=settings.redis_url,
        conformal_method=settings.conformal_method,
        alpha=settings.conformal_alpha,
        eci_eta=settings.eci_eta,
        eci_decay=settings.eci_decay,
        eci_window=settings.eci_window,
        aci_eta=settings.aci_eta,
        pid_eta=settings.pid_eta,
    )


__all__ = [
    "ConformalMethod",
    "OnlineConformalStore",
    "StratumState",
    "build_store_from_settings",
    "stratum_key",
]
