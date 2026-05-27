"""
Training panel: ICCO national yields + CRIG station stub paired with country-mean climate stacks.

ICCO rows are augmented with bootstrap farm-level draws ``N(mean, 0.4 t/ha)`` around each
country-year mean yield. Observed yields are adjusted to pre-biotic targets via
``y_observed / surviving_biotic_fraction`` using :mod:`hazards` priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from hazards.composite import estimate_surviving_biotic_fraction
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES, CLIMATE_IDX, N_CLIMATE_CHANNELS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ICCO_GLOB = _REPO_ROOT / "data" / "external" / "icco_*.csv"
DEFAULT_CRIG_PATH = _REPO_ROOT / "data" / "raw" / "crig_station_yields.csv"

Cohort = Literal["icco", "station"]

# Country centroids (lat, lon) for synthetic country-mean ERA5 when Zarr is absent.
COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "GHA": (6.5, -1.5),
    "CIV": (6.8, -5.5),
    "CMR": (4.5, 9.5),
    "NGA": (6.5, 5.5),
    "ECU": (-0.5, -79.0),
    "IDN": (-0.5, 115.0),
    "PER": (-9.0, -75.0),
    "COL": (4.0, -74.0),
}

# Climatic offsets for country-mean daily stacks (synthetic ERA5 prior).
_COUNTRY_CLIMATE_OFFSETS: dict[str, dict[str, float]] = {
    "GHA": {"tmean": 26.0, "rh_mean": 82.0, "precip": 6.0},
    "CIV": {"tmean": 26.5, "rh_mean": 84.0, "precip": 7.0},
    "CMR": {"tmean": 25.5, "rh_mean": 80.0, "precip": 8.0},
    "NGA": {"tmean": 27.0, "rh_mean": 78.0, "precip": 5.5},
    "ECU": {"tmean": 25.0, "rh_mean": 88.0, "precip": 9.0},
    "IDN": {"tmean": 27.5, "rh_mean": 86.0, "precip": 8.5},
    "PER": {"tmean": 24.5, "rh_mean": 82.0, "precip": 6.5},
    "COL": {"tmean": 26.0, "rh_mean": 84.0, "precip": 7.5},
}


def load_icco_tables(glob_pattern: Path | str | None = None) -> pd.DataFrame:
    """Load and concatenate ``data/external/icco_*.csv`` production tables."""
    pattern = Path(glob_pattern) if glob_pattern else DEFAULT_ICCO_GLOB
    paths = sorted(pattern.parent.glob(pattern.name))
    if not paths:
        raise FileNotFoundError(f"No ICCO CSV files matching {pattern}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"country_iso3", "year", "production_tonnes", "planted_area_ha"}
    if not required.issubset(df.columns):
        raise ValueError(f"ICCO tables missing columns: {required - set(df.columns)}")
    df["yield_t_ha"] = df["production_tonnes"] / df["planted_area_ha"]
    return df


def _synthetic_crig_stations() -> pd.DataFrame:
    """CRIG station yield stub (replace via DVC remote for production calibration)."""
    rows: list[dict[str, Any]] = []
    stations = [
        ("GHA", "tafo", 6.45, -0.58, 0.55),
        ("GHA", "akim", 6.30, -0.90, 0.48),
        ("CIV", "divo", 5.75, -5.36, 0.62),
        ("CIV", "san_pedro", 4.75, -6.88, 0.58),
        ("CMR", "mbalmayo", 3.50, 11.50, 0.52),
    ]
    for iso, name, lat, lon, base_yield in stations:
        for year in range(2015, 2025):
            rows.append(
                {
                    "country_iso3": iso,
                    "station_id": name,
                    "year": year,
                    "lat": lat,
                    "lon": lon,
                    "yield_t_ha": float(base_yield + 0.02 * (year - 2018)),
                }
            )
    return pd.DataFrame(rows)


def load_crig_station_table(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or DEFAULT_CRIG_PATH
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
    else:
        df = _synthetic_crig_stations()
    required = {"country_iso3", "year", "yield_t_ha"}
    if not required.issubset(df.columns):
        raise ValueError(f"CRIG table missing columns: {required - set(df.columns)}")
    return df


def build_country_climate_stack(
    country_iso3: str,
    year: int,
    *,
    sequence_length: int = 365,
    seed: int | None = None,
) -> np.ndarray:
    """
    Country-mean daily climate tensor ``[T, 11]`` (synthetic ERA5 prior).

    When processed ERA5 Zarr is wired, replace this with zonal means at
    :data:`COUNTRY_CENTROIDS`.
    """
    rng = np.random.default_rng(seed if seed is not None else hash((country_iso3, year)) % (2**32))
    iso = str(country_iso3)
    offsets = _COUNTRY_CLIMATE_OFFSETS.get(iso, _COUNTRY_CLIMATE_OFFSETS["GHA"])
    tmean = offsets["tmean"] + rng.normal(0, 0.4, sequence_length).astype(np.float32)
    tmax = tmean + rng.uniform(2.0, 4.0, sequence_length).astype(np.float32)
    tmin = tmean - rng.uniform(2.0, 4.0, sequence_length).astype(np.float32)
    precip = np.clip(
        offsets["precip"] + rng.exponential(2.0, sequence_length),
        0.0,
        None,
    ).astype(np.float32)
    rh = np.clip(offsets["rh_mean"] + rng.normal(0, 3.0, sequence_length), 55.0, 98.0).astype(
        np.float32
    )
    vpd = np.clip(1.2 - 0.01 * (rh - 75.0) + rng.normal(0, 0.1, sequence_length), 0.3, 2.5).astype(
        np.float32
    )
    srad = (14.0 + rng.normal(0, 1.5, sequence_length)).astype(np.float32)
    et0 = (3.5 + 0.05 * (tmean - 25.0) + rng.normal(0, 0.2, sequence_length)).astype(np.float32)
    sm_root = np.clip(0.28 + rng.normal(0, 0.03, sequence_length), 0.08, 0.45).astype(np.float32)
    wind10m = (2.0 + rng.normal(0, 0.3, sequence_length)).astype(np.float32)
    co2 = np.full(sequence_length, 415.0 + 0.5 * (year - 2015), dtype=np.float32)

    stack = np.stack(
        [tmax, tmin, tmean, precip, srad, vpd, et0, sm_root, wind10m, rh, co2],
        axis=-1,
    )
    assert stack.shape == (sequence_length, N_CLIMATE_CHANNELS)
    return stack


def climate_array_to_dataset(climate: np.ndarray, year: int) -> xr.Dataset:
    """``[T, 11]`` numpy stack → daily ``xr.Dataset`` for biotic hazard models."""
    time = pd.date_range(f"{year}-01-01", periods=climate.shape[0], freq="D")
    data_vars = {
        name: ("time", climate[:, CLIMATE_IDX[name]].astype(np.float32))
        for name in CLIMATE_CHANNEL_NAMES
    }
    return xr.Dataset(data_vars, coords={"time": time})


def encode_static_features(
    *,
    yield_t_ha: float,
    country_iso3: str,
    awc_mm: float | None = None,
) -> np.ndarray:
    """Map panel row to :class:`~models.yield_surrogate.YieldSurrogateModel` static vector."""
    awc_defaults = {
        "GHA": 140.0,
        "CIV": 130.0,
        "CMR": 150.0,
        "NGA": 120.0,
        "ECU": 160.0,
        "IDN": 135.0,
    }
    awc = awc_mm if awc_mm is not None else awc_defaults.get(str(country_iso3), 135.0)
    static = np.zeros(10, dtype=np.float32)
    static[0] = awc
    static[1] = 0.35  # sand_frac prior
    static[2] = float(yield_t_ha) / 5.0  # baseline_yield_scaled
    static[3] = 0.0  # intervention_flag
    static[4] = 0.0  # stress_tolerance
    static[5] = 0.25  # clay_frac
    static[6] = 0.5  # soc_norm
    static[7] = 0.55  # ph_norm
    static[8] = 0.4  # treecover_norm
    static[9] = 0.85  # cocoa_prob
    return static


def pre_biotic_yield_target(
    yield_observed_t_ha: float,
    climate: np.ndarray,
    year: int,
    *,
    biotic_static: dict[str, Any] | None = None,
    min_surviving: float = 0.15,
) -> tuple[float, float]:
    """
    Back out climate-driven potential yield from observed yield and biotic priors.

    Returns ``(y_target_pre_biotic, surviving_biotic_fraction)``.
    """
    ds = climate_array_to_dataset(climate, year)
    surviving = estimate_surviving_biotic_fraction(ds, biotic_static)
    surviving = max(float(surviving), min_surviving)
    return float(yield_observed_t_ha) / surviving, surviving


@dataclass(frozen=True)
class PanelRow:
    sample_id: str
    country_iso3: str
    year: int
    cohort: Cohort
    yield_observed_t_ha: float
    yield_target_pre_biotic_t_ha: float
    surviving_biotic_fraction: float
    climate: np.ndarray
    static: np.ndarray


def build_yield_panel(
    *,
    icco_glob: Path | str | None = None,
    crig_path: Path | None = None,
    bootstrap_per_country_year: int = 8,
    augment_sigma_t_ha: float = 0.4,
    sequence_length: int = 365,
    seed: int = 42,
) -> list[PanelRow]:
    """Assemble ICCO-augmented + CRIG station training rows."""
    rng = np.random.default_rng(seed)
    rows: list[PanelRow] = []

    icco = load_icco_tables(icco_glob)
    for _, rec in icco.iterrows():
        iso = str(rec["country_iso3"])
        year = int(rec["year"])
        mean_y = float(rec["yield_t_ha"])
        climate = build_country_climate_stack(iso, year, sequence_length=sequence_length)
        biotic_static = {
            "cssvd_prevalence_pct": 15.0,
            "cssvd_tolerance": 1.0,
            "shade_species": "unshaded",
        }
        for draw in range(bootstrap_per_country_year):
            y_obs = float(mean_y + rng.normal(0.0, augment_sigma_t_ha))
            y_obs = max(y_obs, 0.05)
            y_tgt, surv = pre_biotic_yield_target(y_obs, climate, year, biotic_static=biotic_static)
            static = encode_static_features(yield_t_ha=y_tgt, country_iso3=iso)
            rows.append(
                PanelRow(
                    sample_id=f"icco_{iso}_{year}_{draw}",
                    country_iso3=iso,
                    year=year,
                    cohort="icco",
                    yield_observed_t_ha=y_obs,
                    yield_target_pre_biotic_t_ha=y_tgt,
                    surviving_biotic_fraction=surv,
                    climate=climate,
                    static=static,
                )
            )

    crig = load_crig_station_table(crig_path)
    for _, rec in crig.iterrows():
        iso = str(rec["country_iso3"])
        year = int(rec["year"])
        y_obs = float(rec["yield_t_ha"])
        station_id = str(rec.get("station_id", "station"))
        climate = build_country_climate_stack(
            iso,
            year,
            sequence_length=sequence_length,
            seed=hash((station_id, year)) % (2**32),
        )
        biotic_static = {
            "cssvd_prevalence_pct": 20.0,
            "cssvd_tolerance": 1.0,
            "shade_species": "unshaded",
        }
        y_tgt, surv = pre_biotic_yield_target(y_obs, climate, year, biotic_static=biotic_static)
        static = encode_static_features(yield_t_ha=y_tgt, country_iso3=iso)
        rows.append(
            PanelRow(
                sample_id=f"crig_{station_id}_{year}",
                country_iso3=iso,
                year=year,
                cohort="station",
                yield_observed_t_ha=y_obs,
                yield_target_pre_biotic_t_ha=y_tgt,
                surviving_biotic_fraction=surv,
                climate=climate,
                static=static,
            )
        )

    return rows


class YieldPanelDataset(Dataset[dict[str, torch.Tensor | str]]):
    """PyTorch dataset over :func:`build_yield_panel` rows."""

    def __init__(self, panel_rows: list[PanelRow]) -> None:
        self._rows = panel_rows

    @classmethod
    def from_config(
        cls,
        *,
        icco_glob: Path | str | None = None,
        crig_path: Path | None = None,
        bootstrap_per_country_year: int = 8,
        augment_sigma_t_ha: float = 0.4,
        sequence_length: int = 365,
        seed: int = 42,
    ) -> YieldPanelDataset:
        return cls(
            build_yield_panel(
                icco_glob=icco_glob,
                crig_path=crig_path,
                bootstrap_per_country_year=bootstrap_per_country_year,
                augment_sigma_t_ha=augment_sigma_t_ha,
                sequence_length=sequence_length,
                seed=seed,
            )
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self._rows[idx]
        return {
            "climate": torch.from_numpy(row.climate),  # [T, 11]
            "static": torch.from_numpy(row.static),  # [10]
            "target": torch.tensor(row.yield_target_pre_biotic_t_ha, dtype=torch.float32),
            "yield_observed": torch.tensor(row.yield_observed_t_ha, dtype=torch.float32),
            "country_iso3": row.country_iso3,
            "cohort": row.cohort,
            "year": row.year,
            "sample_id": row.sample_id,
        }


def panel_train_val_split(
    panel_rows: list[PanelRow],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[PanelRow], list[PanelRow]]:
    """Stratified holdout by country (last years to val)."""
    rng = np.random.default_rng(seed)
    by_country: dict[str, list[PanelRow]] = {}
    for row in panel_rows:
        by_country.setdefault(row.country_iso3, []).append(row)
    train: list[PanelRow] = []
    val: list[PanelRow] = []
    for iso, group in by_country.items():
        group = sorted(group, key=lambda r: (r.year, r.sample_id))
        n_val = max(1, int(len(group) * val_fraction))
        val.extend(group[-n_val:])
        train.extend(group[:-n_val])
        _ = iso
    rng.shuffle(train)
    return train, val


__all__ = [
    "COUNTRY_CENTROIDS",
    "PanelRow",
    "YieldPanelDataset",
    "build_country_climate_stack",
    "build_yield_panel",
    "climate_array_to_dataset",
    "encode_static_features",
    "load_crig_station_table",
    "load_icco_tables",
    "panel_train_val_split",
    "pre_biotic_yield_target",
]
