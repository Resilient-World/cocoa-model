"""Tests for ATTRICI subprocess wrapper (GPL boundary, physics helpers)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from counterfactual.attrici_runner import (
    ATTRICI_DISTRIBUTIONS,
    ATTRICIConfig,
    _buck_huss,
    buck_saturation_vapor_pressure_hpa,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ATTRICI_BIN = REPO_ROOT / ".venv-attrici" / "bin" / "attrici"
SHIM = REPO_ROOT / "scripts" / "attrici_cli_shim.py"

# Mengel et al. 2021, GMD 14, 5269 — Table 1 (§3.2); exactly seven variables
MENGEL_2021_TABLE_1: dict[str, tuple[str, str]] = {
    "tas": ("normal", "identity"),
    "tasrange": ("gamma", "log"),
    "tasskew": ("normal", "identity"),
    "pr": ("bernoulli_gamma", "log"),
    "rsds": ("normal", "identity"),
    "sfcwind": ("weibull", "log"),
    "hurs": ("beta", "logit"),
}

_ATTRICI_IMPORT_RE = re.compile(r"^\s*(import attrici|from attrici\b)")


def test_no_direct_attrici_imports() -> None:
    hits: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _ATTRICI_IMPORT_RE.match(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}")
    assert hits == []


def test_distribution_table_matches_mengel_2021() -> None:
    assert ATTRICI_DISTRIBUTIONS == MENGEL_2021_TABLE_1
    assert len(ATTRICI_DISTRIBUTIONS) == 7


def test_buck_1981_vapor_pressure_at_25c() -> None:
    e_s = float(buck_saturation_vapor_pressure_hpa(25.0))
    assert e_s == pytest.approx(31.69, rel=0.005)


def test_huss_derivation_monotonic_in_hurs() -> None:
    tas = xr.DataArray(25.0)
    ps = xr.DataArray(101_325.0)
    # ISIMIP ``hurs`` is percent; 0.3 / 0.6 / 0.9 denote 30 / 60 / 90 % RH
    hurs_values = (30.0, 60.0, 90.0)
    huss_values = [_buck_huss(tas, xr.DataArray(h), ps).item() for h in hurs_values]
    assert huss_values[0] < huss_values[1] < huss_values[2]


def _write_one_cell_toy_obs(path: Path, *, start: str = "2015-01-01", end: str = "2019-12-31") -> tuple[float, float]:
    time = pd.date_range(start, end, freq="D")
    lat, lon = 6.0, -5.0
    n = len(time)
    seasonal = 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 365.25)
    tas_k = 280.0 + seasonal
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), tas_k[:, np.newaxis, np.newaxis])},
        coords={"time": time, "lat": [lat], "lon": [lon]},
    )
    ds.to_netcdf(path)
    return lat, lon


def _write_toy_gmt(path: Path, time: pd.DatetimeIndex) -> None:
    n = len(time)
    gmt = np.linspace(0.2, 0.8, n) + 0.05 * np.sin(2.0 * np.pi * np.arange(n) / 365.25)
    xr.Dataset({"tas": ("time", gmt)}, coords={"time": time}).to_netcdf(path)


@pytest.mark.integration
def test_attrici_subprocess_runs_on_one_cell(tmp_path: Path) -> None:
    if not ATTRICI_BIN.is_file():
        pytest.skip(f"ATTRICI not installed: {ATTRICI_BIN}")

    obs_path = tmp_path / "obs_tas.nc"
    lat, lon = _write_one_cell_toy_obs(obs_path)
    time = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    gmt_path = tmp_path / "gmt.nc"
    _write_toy_gmt(gmt_path, time)

    out_dir = tmp_path / "detrend_out"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        str(SHIM),
        "--attrici-bin",
        str(ATTRICI_BIN),
        "--gmt-file",
        str(gmt_path),
        "--input-file",
        str(obs_path),
        "--output-dir",
        str(out_dir),
        "--variable",
        "tas",
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--modes",
        "4",
        "--solver",
        "scipy",
        "--start-date",
        "2015-01-01",
        "--stop-date",
        "2019-12-31",
        "--overwrite",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr

    out_file = out_dir / "timeseries" / "tas" / f"lat_{lat:g}" / f"ts_lat{lat:g}_lon{lon:g}.nc"
    assert out_file.is_file(), f"missing {out_file}; stderr={result.stderr}"

    out_ds = xr.open_dataset(out_file)
    try:
        assert "time" in out_ds.dims
        assert out_ds.sizes["time"] == len(time)
        assert out_ds.sizes.get("lat", 1) == 1
        assert out_ds.sizes.get("lon", 1) == 1
        assert "cfact" in out_ds or "tas" in out_ds or len(out_ds.data_vars) >= 1
    finally:
        out_ds.close()
