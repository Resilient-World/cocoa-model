"""Probabilistic forecast baselines for CRPSS evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from data.cocoa_exposure import REGIONS
from data.yield_panel import PanelRow
from validation.forecast_scoring import crps_ensemble, crpss

ISO3_TO_REGION: dict[str, str] = {
    "GHA": "ghana",
    "CIV": "civ",
    "CMR": "cameroon",
    "NGA": "nigeria",
    "ECU": "ecuador",
    "IDN": "indonesia",
}


@dataclass(frozen=True)
class StratumKey:
    scenario: str
    horizon_year: int
    region: str

    def as_key(self) -> str:
        return f"{self.scenario}:{self.horizon_year}:{self.region}"


@dataclass
class BaselineForecast:
    """Ensemble members ``(n, m)`` for CRPS."""

    ensemble: np.ndarray


class BaselinePredictor(Protocol):
    def predict(
        self,
        rows: list[PanelRow],
        indices: np.ndarray,
        *,
        stratum: StratumKey,
        rng: np.random.Generator,
    ) -> BaselineForecast: ...


def panel_stratum(row: PanelRow, *, scenario: str = "ssp245") -> StratumKey:
    region = ISO3_TO_REGION.get(row.country_iso3, "ghana")
    return StratumKey(scenario=scenario, horizon_year=int(row.year), region=region)


def group_indices_by_stratum(
    rows: list[PanelRow],
    indices: np.ndarray,
    *,
    scenario: str = "ssp245",
) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for i in indices:
        key = panel_stratum(rows[int(i)], scenario=scenario).as_key()
        buckets.setdefault(key, []).append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}


class ClimatologyBaseline:
    """Historical yield distribution per region-year (bootstrap ensemble)."""

    def __init__(self, *, n_members: int = 50, noise_scale: float = 0.05) -> None:
        self.n_members = n_members
        self.noise_scale = noise_scale

    def predict(
        self,
        rows: list[PanelRow],
        indices: np.ndarray,
        *,
        stratum: StratumKey,
        rng: np.random.Generator,
    ) -> BaselineForecast:
        pool = [
            rows[i].yield_target_pre_biotic_t_ha
            for i in range(len(rows))
            if panel_stratum(rows[i]).region == stratum.region
        ]
        if not pool:
            pool = [rows[int(i)].yield_target_pre_biotic_t_ha for i in indices]
        n = len(indices)
        m = self.n_members
        ens = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            base = rng.choice(pool)
            ens[i] = base + rng.normal(0, self.noise_scale, size=m)
        return BaselineForecast(ensemble=ens)


class PersistenceBaseline:
    """Last calendar year's yield per country (with small spread)."""

    def __init__(self, *, n_members: int = 21, noise_scale: float = 0.03) -> None:
        self.n_members = n_members
        self.noise_scale = noise_scale

    def predict(
        self,
        rows: list[PanelRow],
        indices: np.ndarray,
        *,
        stratum: StratumKey,
        rng: np.random.Generator,
    ) -> BaselineForecast:
        by_country_year: dict[tuple[str, int], float] = {}
        for r in rows:
            by_country_year[(r.country_iso3, r.year)] = r.yield_target_pre_biotic_t_ha
        n = len(indices)
        m = self.n_members
        ens = np.empty((n, m), dtype=np.float64)
        for j, idx in enumerate(indices):
            r = rows[int(idx)]
            prev = by_country_year.get((r.country_iso3, r.year - 1))
            if prev is None:
                prev = r.yield_target_pre_biotic_t_ha
            ens[j] = prev + rng.normal(0, self.noise_scale, size=m)
        return BaselineForecast(ensemble=ens)


class FDPMeanBaseline:
    """Region-mean FDP cocoa probability × yield prior."""

    def __init__(
        self,
        *,
        yield_prior_t_ha: float = 1.2,
        n_members: int = 21,
        noise_scale: float = 0.04,
    ) -> None:
        self.yield_prior_t_ha = yield_prior_t_ha
        self.n_members = n_members
        self.noise_scale = noise_scale

    def _region_prob(self, region: str) -> float:
        if region in REGIONS:
            return 0.85
        return 0.5

    def predict(
        self,
        rows: list[PanelRow],
        indices: np.ndarray,
        *,
        stratum: StratumKey,
        rng: np.random.Generator,
    ) -> BaselineForecast:
        center = self._region_prob(stratum.region) * self.yield_prior_t_ha
        n = len(indices)
        m = self.n_members
        ens = center + rng.normal(0, self.noise_scale, size=(n, m))
        return BaselineForecast(ensemble=ens)


def evaluate_with_baselines(
    observations: np.ndarray,
    model_ensemble: np.ndarray,
    rows: list[PanelRow],
    indices: np.ndarray,
    *,
    scenario: str = "ssp245",
    seed: int = 0,
) -> dict[str, Any]:
    """
    Pooled and per-stratum CRPS / CRPSS vs three baselines.
    """
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    model_ens = np.asarray(model_ensemble, dtype=np.float64)
    if model_ens.ndim == 1:
        model_ens = model_ens.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    baselines: dict[str, BaselinePredictor] = {
        "climatology": ClimatologyBaseline(),
        "persistence": PersistenceBaseline(),
        "fdp_mean": FDPMeanBaseline(),
    }
    crps_model = float(np.nanmean(crps_ensemble(obs, model_ens)))
    result: dict[str, Any] = {
        "crps": crps_model,
        "crpss_climatology": float("nan"),
        "crpss_persistence": float("nan"),
        "crpss_fdp_mean": float("nan"),
        "by_stratum": {},
    }
    for name, predictor in baselines.items():
        base_ens_list: list[np.ndarray] = []
        for idx in indices:
            sk = panel_stratum(rows[int(idx)], scenario=scenario)
            fc = predictor.predict(rows, np.array([idx]), stratum=sk, rng=rng)
            base_ens_list.append(fc.ensemble[0])
        base_ens = np.stack(base_ens_list, axis=0)
        crps_b = float(np.nanmean(crps_ensemble(obs, base_ens)))
        result[f"crpss_{name}"] = crpss(crps_model, crps_b)

    pos_map = {int(indices[k]): k for k in range(len(indices))}
    by_stratum = group_indices_by_stratum(rows, indices, scenario=scenario)
    for key, idx in by_stratum.items():
        pos = np.array([pos_map[int(i)] for i in idx if int(i) in pos_map], dtype=np.int64)
        if pos.size == 0:
            continue
        o = obs[pos]
        m_ens = model_ens[pos]
        sub: dict[str, Any] = {"crps": float(np.nanmean(crps_ensemble(o, m_ens)))}
        parts = key.split(":")
        sk = StratumKey(scenario=parts[0], horizon_year=int(parts[1]), region=parts[2])
        for name, predictor in baselines.items():
            fc = predictor.predict(rows, idx, stratum=sk, rng=rng)
            crps_b = float(np.nanmean(crps_ensemble(o, fc.ensemble)))
            sub[f"crpss_{name}"] = crpss(sub["crps"], crps_b)
        result["by_stratum"][key] = sub
    return result


__all__ = [
    "BaselineForecast",
    "ClimatologyBaseline",
    "FDPMeanBaseline",
    "ISO3_TO_REGION",
    "PersistenceBaseline",
    "StratumKey",
    "evaluate_with_baselines",
    "group_indices_by_stratum",
    "panel_stratum",
]
