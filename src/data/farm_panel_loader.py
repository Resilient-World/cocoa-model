"""Schema-validated farm panel loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.decorators import check_io

from data.farm_panel import load_real_panel as _load_real_panel
from data.farm_panel import load_synthetic_panel as _load_synthetic_panel
from data.schemas import FarmPanelSchema, validate_dataframe


def _to_schema_panel(df: pd.DataFrame, *, treatment_year: int | None = None) -> pd.DataFrame:
    """Map internal farm_panel columns to FarmPanelSchema contract."""
    out = df.copy()
    if "treatment" not in out.columns:
        out["treatment"] = out["received_intervention"].astype(int)
    if "yield_pre" not in out.columns or "yield_post" not in out.columns:
        ty = treatment_year
        if ty is None and "year" in out.columns:
            years = sorted(out["year"].unique())
            ty = years[len(years) // 2] if years else years[0]
        pre_mask = out["year"] < ty if ty is not None else out["treatment"] == 0
        out["yield_pre"] = (
            out.loc[pre_mask].groupby("farm_id")["yield_tonnes_per_ha"].transform("mean")
        )
        out["yield_post"] = (
            out.loc[~pre_mask].groupby("farm_id")["yield_tonnes_per_ha"].transform("mean")
        )
        out["yield_pre"] = out["yield_pre"].fillna(out["yield_tonnes_per_ha"])
        out["yield_post"] = out["yield_post"].fillna(out["yield_tonnes_per_ha"])
    if "cocoa_price_usd" not in out.columns:
        out["cocoa_price_usd"] = 3200.0
    cols = [
        "farm_id",
        "treatment",
        "yield_pre",
        "yield_post",
        "farm_size_ha",
        "lat",
        "lon",
        "cocoa_price_usd",
    ]
    return validate_dataframe(FarmPanelSchema, out[cols])


@check_io(out=FarmPanelSchema)
def load_synthetic_panel_validated(
    *,
    n_farms: int = 5000,
    n_years: int = 8,
    treatment_year: int = 4,
    true_att: float = 0.35,
    seed: int = 42,
    start_calendar_year: int = 2016,
) -> pd.DataFrame:
    """Synthetic panel with Pandera validation."""
    df = _load_synthetic_panel(
        n_farms=n_farms,
        n_years=n_years,
        treatment_year=treatment_year,
        true_att=true_att,
        seed=seed,
        start_calendar_year=start_calendar_year,
    )
    return _to_schema_panel(df, treatment_year=start_calendar_year + treatment_year)


@check_io(out=FarmPanelSchema)
def load_real_panel_validated(parquet_path: Path | str) -> pd.DataFrame:
    """Observed panel from parquet with Pandera validation."""
    df = _load_real_panel(parquet_path)
    return _to_schema_panel(df)


def load_farm_panel(
    parquet_path: Path | str | None = None,
    *,
    synthetic: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """Unified entrypoint for causal / validation pipelines."""
    if synthetic or parquet_path is None:
        return load_synthetic_panel_validated(**kwargs)  # type: ignore[arg-type]
    return load_real_panel_validated(parquet_path)
