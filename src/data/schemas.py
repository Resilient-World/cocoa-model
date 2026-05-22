"""Pandera data contracts for ingestion and panel loaders (strict validation)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# Canonical ERA5 internal names ↔ regulator aliases
ERA5_COLUMN_ALIASES: dict[str, str] = {
    "t2m_max": "tmax",
    "t2m_min": "tmin",
    "tp": "precip",
    "ssrd": "srad",
    "u10": "wind10m",
    "v10": "wind10m",
}

ERA5_REQUIRED_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "date",
    "tmax",
    "tmin",
    "precip",
    "srad",
    "wind10m",
)


class ERA5DailySchema(pa.DataFrameModel):
    """Daily ERA5-Land stack (point or gridded sample rows)."""

    lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    lon: Series[float] = pa.Field(ge=-180.0, le=180.0)
    date: Series[Any]
    tmax: Series[float] = pa.Field(ge=-50.0, le=60.0)
    tmin: Series[float] = pa.Field(ge=-50.0, le=60.0)
    precip: Series[float] = pa.Field(ge=0.0, le=500.0)
    srad: Series[float] = pa.Field(ge=0.0, le=40.0)
    wind10m: Series[float] = pa.Field(ge=0.0, le=50.0, nullable=True)

    class Config:
        strict = False
        coerce = True

    @pa.dataframe_check
    def tmax_ge_tmin(cls, df: pd.DataFrame) -> Series[bool]:
        return df["tmax"] >= df["tmin"]


class FDPProbabilitySchema(pa.DataFrameModel):
    """Point-level FDP cocoa probability samples."""

    lon: Series[float] = pa.Field(ge=-180.0, le=180.0)
    lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    probability: Series[float] = pa.Field(ge=0.0, le=1.0)
    year: Series[int] = pa.Field(isin=[2020, 2023])

    class Config:
        strict = True
        coerce = True


class FarmPanelSchema(pa.DataFrameModel):
    """Farm-year panel for causal / conformal evaluation."""

    farm_id: Series[str]
    treatment: Series[int] = pa.Field(isin=[0, 1])
    yield_pre: Series[float] = pa.Field(ge=0.0)
    yield_post: Series[float] = pa.Field(ge=0.0)
    farm_size_ha: Series[float] = pa.Field(gt=0.0)
    lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    lon: Series[float] = pa.Field(ge=-180.0, le=180.0)
    cocoa_price_usd: Series[float] = pa.Field(gt=0.0, nullable=True)

    class Config:
        strict = False
        coerce = True


class CMIP6ScenarioSchema(pa.DataFrameModel):
    """CMIP6 scenario ensemble index rows."""

    scenario: Series[str] = pa.Field(isin=["ssp245", "ssp585"])
    horizon_year: Series[int] = pa.Field(isin=[2030, 2050, 2080])
    variable: Series[str]
    model: Series[str]
    lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    lon: Series[float] = pa.Field(ge=-180.0, le=180.0)
    time: Series[Any]

    class Config:
        strict = False
        coerce = True


class SentinelTileManifestSchema(pa.DataFrameModel):
    """Sentinel composite export manifest rows."""

    region: Series[str]
    start_date: Series[str]
    end_date: Series[str]
    export_path: Series[str] = pa.Field(nullable=True)
    ndvi_min: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=True)
    ndvi_max: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=True)

    class Config:
        strict = False
        coerce = True


def normalize_era5_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map regulator aliases to internal ERA5 names."""
    out = df.copy()
    for alias, canonical in ERA5_COLUMN_ALIASES.items():
        if alias in out.columns and canonical not in out.columns:
            out[canonical] = out[alias]
    missing = [c for c in ERA5_REQUIRED_COLUMNS if c not in out.columns]
    if missing and "wind10m" in missing and "u10" in out.columns:
        out["wind10m"] = out["u10"]
        missing = [c for c in ERA5_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"ERA5 frame missing columns after alias map: {missing}")
    return out


def validate_dataframe(schema: type[pa.DataFrameModel], df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return DataFrame; raise ValueError with actionable detail."""
    try:
        if schema is ERA5DailySchema:
            df = normalize_era5_columns(df)
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.head(3).to_dict(orient="records")
        raise ValueError(
            f"Schema validation failed for {schema.__name__}: {exc.message}; "
            f"examples={cases}"
        ) from exc


def zarr_to_daily_df(ds: Any, *, max_points: int = 500) -> pd.DataFrame:
    """Sample a Zarr ERA5 dataset to a flat daily DataFrame for schema checks."""
    import xarray as xr

    if not isinstance(ds, xr.Dataset):
        ds = xr.open_zarr(ds, consolidated=True)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values.ravel()[: max_points // 10 + 1]
    lons = ds[lon_name].values.ravel()[: max_points // 10 + 1]
    times = pd.to_datetime(ds["time"].values if "time" in ds.coords else ds.indexes.get("time", []))
    rows: list[dict[str, Any]] = []
    for la in lats[:3]:
        for lo in lons[:3]:
            for t in times[: min(30, len(times))]:
                row: dict[str, Any] = {"lat": float(la), "lon": float(lo), "date": t}
                for var in ("tmax", "tmin", "precip", "srad", "wind10m"):
                    if var in ds:
                        sel = ds[var].sel({lat_name: la, lon_name: lo, "time": t}, method="nearest")
                        row[var] = float(sel.values)
                if "tmax" in row and "tmin" in row:
                    rows.append(row)
                if len(rows) >= max_points:
                    break
    return pd.DataFrame(rows)
