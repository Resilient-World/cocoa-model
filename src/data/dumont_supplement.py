"""
Dumont et al. (2025) CSSVD plot supplement ingest and feature join.

Expected supplement: ~2,847 plots with coordinates and survival outcomes
(duration + event, or 12-month incidence flag).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from data.cssvd_landscape_features import (
    LandscapeFeatureRow,
    build_landscape_feature_row,
    landscape_features_cache_path,
)
from hazards.cssvd_landscape import (
    HORIZON_MONTHS,
    STRAIN_PREFIX,
    feature_dict_from_row,
    features_to_dataframe,
)

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPLEMENT_PATH = _REPO_ROOT / "data" / "external" / "dumont_cssvd_plots.csv"
DEFAULT_SYNTHETIC_PATH = _REPO_ROOT / "data" / "external" / "dumont_cssvd_plots_synthetic.csv"

# Column aliases after normalization
_LAT_ALIASES = ("lat", "latitude", "plot_lat", "y")
_LON_ALIASES = ("lon", "longitude", "plot_lon", "x")
_DURATION_ALIASES = ("duration", "time", "months", "followup_months", "t")
_EVENT_ALIASES = ("event", "status", "incident", "cssvd_event", "incidence")
_INCIDENCE12_ALIASES = ("incidence_12mo", "incidence_12m", "cssvd_12mo", "event_12mo")
_ID_ALIASES = ("plot_id", "id", "plot", "farm_id")
_COUNTRY_ALIASES = ("country", "nation", "iso")


def _first_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def normalize_dumont_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map supplement columns to canonical names."""
    out = df.copy()
    lat_col = _first_column(out, _LAT_ALIASES)
    lon_col = _first_column(out, _LON_ALIASES)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Supplement must include latitude/longitude columns; got {list(df.columns)}"
        )
    out = out.rename(columns={lat_col: "lat", lon_col: "lon"})

    dur_col = _first_column(out, _DURATION_ALIASES)
    ev_col = _first_column(out, _EVENT_ALIASES)
    inc_col = _first_column(out, _INCIDENCE12_ALIASES)

    if dur_col and ev_col:
        out = out.rename(columns={dur_col: "duration", ev_col: "event"})
        out["event"] = out["event"].astype(int).clip(0, 1)
        out["duration"] = out["duration"].astype(float).clip(lower=0.1)
    elif inc_col:
        out = out.rename(columns={inc_col: "incidence_12mo"})
        out["event"] = out["incidence_12mo"].astype(int).clip(0, 1)
        out["duration"] = np.where(out["event"] == 1, HORIZON_MONTHS, HORIZON_MONTHS)
    else:
        raise ValueError(
            "Supplement needs (duration, event) or 12-month incidence column; "
            f"columns={list(df.columns)}"
        )

    id_col = _first_column(out, _ID_ALIASES)
    if id_col:
        out = out.rename(columns={id_col: "plot_id"})
    else:
        out["plot_id"] = np.arange(len(out), dtype=int)

    country_col = _first_column(out, _COUNTRY_ALIASES)
    if country_col:
        out = out.rename(columns={country_col: "country"})

    return out


def load_dumont_plots(path: Path | str | None = None) -> pd.DataFrame:
    """Load and normalize Dumont supplement CSV."""
    p = Path(path or DEFAULT_SUPPLEMENT_PATH)
    if not p.is_file():
        raise FileNotFoundError(
            f"Dumont supplement not found: {p}. "
            f"Place journal supplement at {DEFAULT_SUPPLEMENT_PATH} or pass --supplement."
        )
    df = pd.read_csv(p)
    return normalize_dumont_columns(df)


def generate_synthetic_supplement(
    n_plots: int = 500,
    *,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Synthetic plot table for CI (Ghana/CI-like coordinates)."""
    rng = np.random.default_rng(seed)
    n_civ = n_plots // 2
    lats = np.concatenate(
        [
            rng.uniform(5.0, 8.5, n_civ),
            rng.uniform(5.0, 8.0, n_plots - n_civ),
        ]
    )
    lons = np.concatenate(
        [
            rng.uniform(-8.0, -3.0, n_civ),
            rng.uniform(-2.5, 0.8, n_plots - n_civ),
        ]
    )
    incidence = rng.binomial(1, 0.25, n_plots)
    # Variable follow-up for survival (avoid all-censored bootstrap resamples).
    duration = np.where(
        incidence == 1,
        rng.uniform(3.0, HORIZON_MONTHS, n_plots),
        rng.uniform(HORIZON_MONTHS, HORIZON_MONTHS * 1.5, n_plots),
    )
    df = pd.DataFrame(
        {
            "plot_id": np.arange(n_plots),
            "lat": lats,
            "lon": lons,
            "country": ["CI"] * n_civ + ["GH"] * (n_plots - n_civ),
            "incidence_12mo": incidence,
            "event": incidence,
            "duration": duration,
        }
    )
    out_path = Path(output_path or DEFAULT_SYNTHETIC_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _features_from_cache(
    cache: pd.DataFrame,
    lat: float,
    lon: float,
) -> dict[str, float] | None:
    if cache.empty:
        return None
    sub = cache[
        (np.isclose(cache["lat"], lat, rtol=0, atol=1e-4))
        & (np.isclose(cache["lon"], lon, rtol=0, atol=1e-4))
    ]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return feature_dict_from_row(
        LandscapeFeatureRow(
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            year=int(row.get("year", 2023)),
            cocoa_probability_local=float(row["cocoa_probability_local"]),
            non_cocoa_buffer_500m=float(row["non_cocoa_buffer_500m"]),
            canopy_fragmentation_index=float(row["canopy_fragmentation_index"]),
            extreme_precip_5day_count_yr=int(row["extreme_precip_5day_count_yr"]),
            dtr_growing_season=float(row["dtr_growing_season"]),
            strain_region=str(row["strain_region"]),  # type: ignore[arg-type]
        )
    )


def _strain_from_dummy_row(r: pd.Series) -> str:
    for s in ("1A", "1B", "1C"):
        if float(r.get(f"{STRAIN_PREFIX}{s}", 0)) >= 0.5:
            return s
    return "2"


def join_exposure_features(
    plots: pd.DataFrame,
    year: int,
    *,
    cache_path: Path | str | None = None,
    use_gee: bool = False,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Join Dumont plots to landscape covariates; write/read parquet cache.
    """
    from data.cssvd_strain_atlas import lookup_strain_region

    cache_p = Path(cache_path or landscape_features_cache_path(year))
    cache_df = pd.DataFrame()
    if cache_p.is_file() and not refresh_cache:
        cache_df = pd.read_parquet(cache_p)

    feature_rows: list[dict[str, float]] = []

    for _, row in plots.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        cached = _features_from_cache(cache_df, lat, lon) if not refresh_cache else None
        if cached is not None:
            feature_rows.append(cached)
            continue
        if use_gee:
            lf = build_landscape_feature_row(lat, lon, year, use_gee_climate=True)
        else:
            rng = np.random.default_rng(int(abs(hash((round(lat, 3), round(lon, 3)))) % (2**32)))
            lf = LandscapeFeatureRow(
                lat=lat,
                lon=lon,
                year=year,
                cocoa_probability_local=float(
                    row.get("cocoa_probability_local", 0.55 + 0.2 * rng.random())
                ),
                non_cocoa_buffer_500m=float(
                    row.get("non_cocoa_buffer_500m", 0.35 + 0.4 * rng.random())
                ),
                canopy_fragmentation_index=float(
                    row.get("canopy_fragmentation_index", 0.8 + rng.random())
                ),
                extreme_precip_5day_count_yr=int(
                    row.get("extreme_precip_5day_count_yr", rng.integers(2, 35))
                ),
                dtr_growing_season=float(row.get("dtr_growing_season", 6.0 + 4.0 * rng.random())),
                strain_region=lookup_strain_region(lat, lon),
            )
        feature_rows.append(feature_dict_from_row(lf))

    X = features_to_dataframe(feature_rows)
    out = pd.concat([plots.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    if refresh_cache or not cache_p.is_file():
        cache_out = out[
            [
                "lat",
                "lon",
                "cocoa_probability_local",
                "non_cocoa_buffer_500m",
                "canopy_fragmentation_index",
                "extreme_precip_5day_count_yr",
                "dtr_growing_season",
            ]
        ].copy()
        cache_out["year"] = year
        cache_out["strain_region"] = [_strain_from_dummy_row(out.loc[i]) for i in range(len(out))]
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_out.to_parquet(cache_p, index=False)

    return out


__all__ = [
    "DEFAULT_SUPPLEMENT_PATH",
    "DEFAULT_SYNTHETIC_PATH",
    "generate_synthetic_supplement",
    "join_exposure_features",
    "load_dumont_plots",
    "normalize_dumont_columns",
]
