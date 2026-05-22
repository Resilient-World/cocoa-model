"""Build CMIP6 ensemble index rows for Pandera validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import xarray as xr

from data.schemas import CMIP6ScenarioSchema, validate_dataframe


def ensemble_to_index_dataframe(
    ds: xr.Dataset,
    *,
    horizon_year: int = 2030,
) -> pd.DataFrame:
    """Flatten ensemble coordinates to schema rows (sampled)."""
    scenarios = [str(s) for s in ds.coords.get("scenario", ["ssp245"]).values]
    models = [str(m) for m in ds.coords.get("model", ["ACCESS-CM2"]).values]
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values.ravel()[:5]
    lons = ds[lon_name].values.ravel()[:5]
    times = pd.to_datetime(ds["time"].values[:3] if "time" in ds.coords else ["2020-01-01"])
    rows: list[dict[str, Any]] = []
    for sc in scenarios[:2]:
        if sc not in ("ssp245", "ssp585"):
            continue
        for mo in models[:2]:
            for la in lats:
                for lo in lons:
                    for t in times:
                        rows.append(
                            {
                                "scenario": sc,
                                "horizon_year": horizon_year,
                                "variable": "tas",
                                "model": mo,
                                "lat": float(la),
                                "lon": float(lo),
                                "time": t,
                            }
                        )
    return pd.DataFrame(rows)


def validate_cmip6_dataset(ds: xr.Dataset, *, horizon_year: int = 2030) -> pd.DataFrame:
    """Validate CMIP6 index sample against :class:`~data.schemas.CMIP6ScenarioSchema`."""
    df = ensemble_to_index_dataframe(ds, horizon_year=horizon_year)
    return validate_dataframe(CMIP6ScenarioSchema, df)
