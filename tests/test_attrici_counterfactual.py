"""Tests for subprocess ATTRICI counterfactual builder (GPL boundary)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.attrici_counterfactual import (
    ATTRICICounterfactual,
    RegionBounds,
    TimeRange,
    normalize_variables,
)
from data.attrici_fast_detrend import FastATTRICICounterfactual, recompute_derived_counterfactuals


def test_normalize_isimip_aliases() -> None:
    norm = normalize_variables(["tas", "pr", "hurs"])
    assert "precip" in norm and "rh_mean" in norm
    assert "tmax" in norm and "tmin" in norm


def test_cache_key_deterministic(tmp_path: Path) -> None:
    factual = tmp_path / "era5.zarr"
    gmt = tmp_path / "gmt.nc"
    gmt.touch()
    _write_minimal_zarr(factual)
    model = ATTRICICounterfactual(factual, gmt_file=gmt, cache_dir=tmp_path / "cache")
    region = RegionBounds(4.0, 8.5, -8.5, -2.5)
    tr = TimeRange(1980, 2024)
    k1 = model.cache_key(["tmax", "precip"], region=region, time_range=tr)
    k2 = model.cache_key(["precip", "tmax"], region=region, time_range=tr)
    assert k1 == k2
    assert model.cached_zarr_path(["tmax", "precip"], region=region, time_range=tr).name == f"cf_{k1}.zarr"


def test_build_uses_cache_without_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factual = tmp_path / "era5.zarr"
    gmt = tmp_path / "gmt.nc"
    xr.Dataset({"tas": ("time", [0.1])}, coords={"time": [0]}).to_netcdf(gmt)
    _write_minimal_zarr(factual)

    model = ATTRICICounterfactual(factual, gmt_file=gmt, cache_dir=tmp_path / "cache")
    out = model.cached_zarr_path(["tmax"], region=RegionBounds(4, 8, -8, -2))
    out.mkdir(parents=True)
    (out / ".zmetadata").write_text("{}")

    mock_run = MagicMock()
    monkeypatch.setattr("counterfactual.attrici_runner.ATTRICIRunner.run", mock_run)

    path = model.build_counterfactual_zarr(["tmax"], region=RegionBounds(4, 8, -8, -2))
    assert path == out
    mock_run.assert_not_called()


def test_build_invokes_runner_when_uncached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factual = tmp_path / "era5.zarr"
    gmt = tmp_path / "gmt.nc"
    xr.Dataset({"tas": ("time", [0.1])}, coords={"time": [0]}).to_netcdf(gmt)
    _write_minimal_zarr(factual)

    model = ATTRICICounterfactual(factual, gmt_file=gmt, cache_dir=tmp_path / "cache")
    target = model.cached_zarr_path(["tmax", "precip"])

    def _fake_run(
        self,
        factual_zarr: Path,
        variables,
        output_zarr: Path,
        overwrite: bool = False,
    ) -> Path:
        output_zarr.mkdir(parents=True, exist_ok=True)
        time = pd.date_range("2020-01-01", periods=10, freq="D")
        for var in variables:
            ds = xr.Dataset(
                {var: (("time", "lat", "lon"), np.ones((10, 1, 1)))},
                coords={"time": time, "lat": [6.0], "lon": [-3.0]},
            )
            ds.to_zarr(output_zarr, group=var, mode="w" if var == variables[0] else "a")
        return output_zarr

    monkeypatch.setattr("counterfactual.attrici_runner.ATTRICIRunner.run", _fake_run)
    monkeypatch.setattr(
        ATTRICICounterfactual,
        "_subset_factual",
        lambda self, variables, region=None, time_range=None, out_zarr=None: out_zarr or factual,
    )
    monkeypatch.setattr(
        ATTRICICounterfactual,
        "_finalize_merged_store",
        lambda self, factual_subset, cf_zarr, variables: None,
    )

    path = model.build_counterfactual_zarr(["tmax", "precip"], overwrite=True)
    assert path == target


def _write_minimal_zarr(path: Path) -> None:
    time = pd.date_range("2020-01-01", periods=30, freq="D")
    ds = xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), np.full((30, 2, 2), 30.0)),
            "tmin": (("time", "lat", "lon"), np.full((30, 2, 2), 22.0)),
            "precip": (("time", "lat", "lon"), np.full((30, 2, 2), 5.0)),
        },
        coords={"time": time, "lat": [5.0, 6.0], "lon": [-4.0, -3.0]},
    )
    ds.to_zarr(path, mode="w")


# --- fast detrender regression tests (no GPLv3 import) ---


def _synthetic_warming_dataset(n_years: int = 40, seed: int = 0) -> tuple[xr.Dataset, pd.Series]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("1980-01-01", periods=n_years * 365, freq="D")
    years = days.year.to_numpy()
    gmt_annual = pd.Series(np.linspace(0.0, 1.5, n_years), index=range(1980, 1980 + n_years))
    gmt_daily = gmt_annual.reindex(years).to_numpy()
    doy = days.dayofyear.to_numpy()
    seasonal = 26 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
    tmax = seasonal + 1.2 * gmt_daily + rng.normal(0, 1.0, len(days))
    precip = np.maximum(0, rng.gamma(0.5, 4, len(days)) - 0.2 * gmt_daily)
    ds = xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), tmax[:, None, None]),
            "precip": (("time", "lat", "lon"), precip[:, None, None]),
        },
        coords={"time": days, "lat": [6.5], "lon": [-1.2]},
    )
    return ds, gmt_annual


def test_fast_counterfactual_removes_trend() -> None:
    ds, gmt = _synthetic_warming_dataset(n_years=8)
    cf = FastATTRICICounterfactual(gmt, variables=("tmax", "precip")).fit_transform(ds)
    fac_trend = np.polyfit(np.arange(len(ds.time)), ds["tmax"].values.ravel(), 1)[0]
    cf_trend = np.polyfit(np.arange(len(cf.time)), cf["tmax_cf"].values.ravel(), 1)[0]
    assert abs(cf_trend) < 0.6 * abs(fac_trend) + 1e-6


def test_recompute_derived_skips_when_inputs_missing() -> None:
    ds, gmt = _synthetic_warming_dataset(n_years=5)
    cf = FastATTRICICounterfactual(gmt, variables=("tmax",)).fit_transform(ds)
    out = recompute_derived_counterfactuals(cf)
    assert "et0_cf" not in out.data_vars
