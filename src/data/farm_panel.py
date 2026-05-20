"""
Farm-year panel for causal evaluation (PSM, AIPW, DiD).

Supports synthetic panels with known ATT and real parquet panels joined with
ERA5 climate aggregates and biotic loss fractions from :mod:`hazards`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from data.yield_panel import build_country_climate_stack, climate_array_to_dataset
from hazards.composite import apply_biotic_losses
from models.yield_surrogate import cohort_phase_from_age

_REPO_ROOT = Path(__file__).resolve().parents[2]

FARM_PANEL_COLUMNS: tuple[str, ...] = (
    "farm_id",
    "year",
    "lat",
    "lon",
    "received_intervention",
    "intervention_type",
    "yield_tonnes_per_ha",
    "farm_size_ha",
    "baseline_yield",
    "soil_quality_index",
    "historical_rainfall",
    "tree_age_years",
    "cssvd_prevalence_pct",
    "shade_species",
)

CLIMATE_AGG_VARS: tuple[str, ...] = (
    "tmean_annual",
    "precip_annual_mm",
    "rh_mean_annual",
    "vpd_mean_annual",
    "sm_root_mean",
)

BIOTIC_LOSS_COLS: tuple[str, ...] = (
    "biotic_black_pod_loss",
    "biotic_cssvd_loss",
    "biotic_mirids_loss",
    "biotic_total_loss_fraction",
    "biotic_surviving_fraction",
)

PSM_COVARIATE_COLS: tuple[str, ...] = (
    "soil_quality_index",
    "historical_rainfall",
    "tree_age_years",
    "cssvd_prevalence_pct",
    "farm_size_ha",
    "tmean_annual",
    "precip_annual_mm",
    "rh_mean_annual",
)


def _validate_panel_columns(df: pd.DataFrame) -> None:
    missing = set(FARM_PANEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")


def load_synthetic_panel(
    *,
    n_farms: int = 5000,
    n_years: int = 8,
    treatment_year: int = 4,
    true_att: float = 0.35,
    seed: int = 42,
    start_calendar_year: int = 2016,
) -> pd.DataFrame:
    """
    Balanced farm-year panel with confounded treatment and known post-period ATT.

    ``treatment_year`` is a 0-based index into the ``n_years`` window: years
    ``< treatment_year`` are pre-intervention; ``>= treatment_year`` receive
    ``true_att`` for treated farms only.
    """
    if treatment_year < 1 or treatment_year >= n_years:
        raise ValueError(f"treatment_year must be in [1, {n_years - 1}]")
    rng = np.random.default_rng(seed)
    years = np.arange(start_calendar_year, start_calendar_year + n_years)

    farm_ids = [f"farm_{i:05d}" for i in range(n_farms)]
    lats = rng.uniform(4.0, 8.0, n_farms)
    lons = rng.uniform(-8.0, -2.0, n_farms)
    farm_size_ha = rng.uniform(2.0, 12.0, n_farms)
    soil_quality_index = rng.uniform(0.2, 0.95, n_farms)
    historical_rainfall = rng.normal(1200.0, 180.0, n_farms)
    baseline_yield = (
        0.15 * farm_size_ha
        + 1.2 * soil_quality_index
        + 0.0004 * historical_rainfall
        + rng.normal(0.0, 0.25, n_farms)
    )
    tree_age_years = rng.integers(6, 28, n_farms).astype(float)
    cssvd_prevalence_pct = rng.uniform(5.0, 35.0, n_farms)

    logit = (
        -0.8
        + 0.05 * farm_size_ha
        + 1.0 * soil_quality_index
        + 0.0006 * historical_rainfall
        - 0.01 * tree_age_years
    )
    ever_treated = (rng.random(n_farms) < (1.0 / (1.0 + np.exp(-logit)))).astype(int)

    rows: list[dict[str, Any]] = []
    for i, farm_id in enumerate(farm_ids):
        for y_idx, year in enumerate(years):
            post = y_idx >= treatment_year
            treated_now = int(ever_treated[i] == 1 and post)
            yield_t = (
                baseline_yield[i]
                + 0.03 * (y_idx - treatment_year)
                + true_att * treated_now
                + rng.normal(0.0, 0.12)
            )
            rows.append(
                {
                    "farm_id": farm_id,
                    "year": int(year),
                    "lat": float(lats[i]),
                    "lon": float(lons[i]),
                    "received_intervention": treated_now,
                    "intervention_type": "shade_trees" if treated_now else "none",
                    "yield_tonnes_per_ha": float(max(yield_t, 0.05)),
                    "farm_size_ha": float(farm_size_ha[i]),
                    "baseline_yield": float(baseline_yield[i]),
                    "soil_quality_index": float(soil_quality_index[i]),
                    "historical_rainfall": float(historical_rainfall[i]),
                    "tree_age_years": float(tree_age_years[i]),
                    "cssvd_prevalence_pct": float(cssvd_prevalence_pct[i]),
                    "shade_species": "khaya_ivorensis" if treated_now else "unshaded",
                }
            )

    panel = pd.DataFrame(rows)
    panel["ever_treated"] = panel["farm_id"].map(
        dict(zip(farm_ids, ever_treated, strict=True))
    )
    return panel


def load_real_panel(parquet_path: Path | str) -> pd.DataFrame:
    """Load an observed farm-year panel from parquet."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Farm panel not found: {path}")
    df = pd.read_parquet(path)
    _validate_panel_columns(df)
    return df


def _synthetic_annual_climate(lat: float, lon: float, year: int) -> dict[str, float]:
    """Deterministic annual aggregates when ERA5 Zarr is unavailable."""
    seed = hash((round(lat, 3), round(lon, 3), year)) % (2**32)
    rng = np.random.default_rng(seed)
    stack = build_country_climate_stack("GHA", year, sequence_length=365)
    return {
        "tmean_annual": float(stack[:, 2].mean()),
        "precip_annual_mm": float(stack[:, 3].sum()),
        "rh_mean_annual": float(stack[:, 9].mean()),
        "vpd_mean_annual": float(stack[:, 5].mean()),
        "sm_root_mean": float(stack[:, 7].mean()),
    }


def _annual_climate_from_zarr(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
) -> dict[str, float]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    if "time" in point.dims:
        t = point["time"]
        if hasattr(t.dt, "year"):
            point = point.sel(time=t.dt.year == year)
        if int(point.sizes.get("time", 0)) == 0:
            point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    out: dict[str, float] = {}
    if "tmean" in point:
        out["tmean_annual"] = float(point["tmean"].mean().values)
    if "precip" in point:
        out["precip_annual_mm"] = float(point["precip"].sum().values)
    if "rh_mean" in point:
        out["rh_mean_annual"] = float(point["rh_mean"].mean().values)
    if "vpd" in point or "vpd_mean" in point:
        vname = "vpd_mean" if "vpd_mean" in point else "vpd"
        out["vpd_mean_annual"] = float(point[vname].mean().values)
    if "sm_root" in point:
        out["sm_root_mean"] = float(point["sm_root"].mean().values)
    return out


def join_climate(
    panel_df: pd.DataFrame,
    era5_zarr_path: Path | str | None = None,
) -> pd.DataFrame:
    """Attach annual climate aggregates per farm-year row."""
    out = panel_df.copy()
    zarr_path = Path(era5_zarr_path) if era5_zarr_path else None
    ds: xr.Dataset | None = None
    if zarr_path is not None and zarr_path.is_dir():
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception:
            ds = None

    records: list[dict[str, float]] = []
    for row in out.itertuples(index=False):
        year = int(row.year)
        lat = float(row.lat)
        lon = float(row.lon)
        if ds is not None:
            try:
                agg = _annual_climate_from_zarr(ds, lat, lon, year)
            except Exception:
                agg = _synthetic_annual_climate(lat, lon, year)
        else:
            agg = _synthetic_annual_climate(lat, lon, year)
        for var in CLIMATE_AGG_VARS:
            agg.setdefault(var, _synthetic_annual_climate(lat, lon, year).get(var, 0.0))
        records.append(agg)

    climate_df = pd.DataFrame(records, index=out.index)
    return pd.concat([out, climate_df], axis=1)


def join_biotic(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Attach modeled biotic loss fractions (black pod, CSSVD, mirids) per row."""
    out = panel_df.copy()
    bp_loss: list[float] = []
    cssvd_loss: list[float] = []
    mirid_loss: list[float] = []
    total_loss: list[float] = []
    surviving: list[float] = []

    for row in out.itertuples(index=False):
        year = int(row.year)
        stack = build_country_climate_stack("GHA", year, sequence_length=365)
        ds = climate_array_to_dataset(stack, year)
        static_features = {
            "cssvd_prevalence_pct": float(getattr(row, "cssvd_prevalence_pct", 15.0)),
            "cssvd_tolerance": 1.0,
            "shade_species": str(getattr(row, "shade_species", "unshaded")),
        }
        result = apply_biotic_losses(1.0, ds, static_features)
        attr = result["loss_attribution"]
        bp_loss.append(attr["black_pod"])
        cssvd_loss.append(attr["cssvd"])
        mirid_loss.append(attr["mirids"])
        total_loss.append(result["total_loss_fraction"])
        surviving.append(result["surviving_fraction"])

    out["biotic_black_pod_loss"] = bp_loss
    out["biotic_cssvd_loss"] = cssvd_loss
    out["biotic_mirids_loss"] = mirid_loss
    out["biotic_total_loss_fraction"] = total_loss
    out["biotic_surviving_fraction"] = surviving
    out["cohort_phase"] = [
        cohort_phase_from_age(float(a)) for a in out["tree_age_years"].to_numpy()
    ]
    return out


def treatment_year_index(panel_df: pd.DataFrame) -> int:
    """Infer 0-based treatment split from first treated farm-year."""
    treated = panel_df.loc[panel_df["received_intervention"] == 1]
    if treated.empty:
        raise ValueError("No treated farm-years; cannot infer treatment_year")
    first_year = int(treated["year"].min())
    years = sorted(panel_df["year"].unique())
    return int(years.index(first_year))


def farm_level_snapshot(
    panel_df: pd.DataFrame,
    *,
    treatment_year: int | None = None,
) -> pd.DataFrame:
    """
    Collapse panel to one row per farm for PSM / AIPW.

    ``received_intervention`` becomes ever-treated; outcome is mean post-period yield.
    """
    if treatment_year is None:
        treatment_year = treatment_year_index(panel_df)

    years = sorted(panel_df["year"].unique())
    if treatment_year < 0 or treatment_year >= len(years):
        raise ValueError(f"treatment_year index {treatment_year} out of range")
    split_year = years[treatment_year]

    meta = panel_df.groupby("farm_id", as_index=False).first()
    ever = panel_df.groupby("farm_id")["received_intervention"].max().rename("received_intervention")
    pre = (
        panel_df.loc[panel_df["year"] < split_year]
        .groupby("farm_id")["yield_tonnes_per_ha"]
        .mean()
        .rename("yield_pre_intervention")
    )
    post = (
        panel_df.loc[panel_df["year"] >= split_year]
        .groupby("farm_id")["yield_tonnes_per_ha"]
        .mean()
        .rename("yield_post_intervention")
    )

    snap = meta.set_index("farm_id")
    snap["received_intervention"] = ever
    snap["yield_pre_intervention"] = pre
    snap["yield_post_intervention"] = post
    snap["yield_tonnes_per_ha"] = post
    return snap.reset_index()


def attach_pre_post_to_matched(
    matched_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure yield pre/post columns are present on PSM output (by ``farm_id``)."""
    out = matched_df.copy()
    needed = ("yield_pre_intervention", "yield_post_intervention")
    if all(c in out.columns for c in needed):
        return out
    cols = ["farm_id", *needed]
    merged = out.drop(columns=[c for c in needed if c in out.columns], errors="ignore").merge(
        snapshot_df[[c for c in cols if c in snapshot_df.columns]],
        on="farm_id",
        how="left",
    )
    if merged[list(needed)].isna().any().any():
        raise ValueError("Matched farms missing pre/post yields in snapshot")
    return merged


__all__ = [
    "BIOTIC_LOSS_COLS",
    "CLIMATE_AGG_VARS",
    "FARM_PANEL_COLUMNS",
    "PSM_COVARIATE_COLS",
    "attach_pre_post_to_matched",
    "farm_level_snapshot",
    "join_biotic",
    "join_climate",
    "load_real_panel",
    "load_synthetic_panel",
    "treatment_year_index",
]
