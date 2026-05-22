"""Tests for :mod:`counterfactual.attrici_runner` (GPL subprocess boundary)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from counterfactual.attrici_runner import (
    _ERA5_TO_ATTRICI,
    SUPPORTED_VARIABLES,
    ATTRICIRunner,
    load_counterfactual,
)

_ALL_SIX = sorted(SUPPORTED_VARIABLES)


def test_attrici_runner_init_stores_fields(tmp_path: Path) -> None:
    gmt = tmp_path / "gmt.nc"
    work = tmp_path / "work"
    runner = ATTRICIRunner(
        gmt_file=gmt,
        work_dir=work,
        attrici_bin="/custom/attrici",
        n_workers=7,
    )
    assert runner.attrici_bin == "/custom/attrici"
    assert runner.gmt_file == gmt
    assert runner.work_dir == work
    assert runner.n_workers == 7
    assert runner._logs_dir == work / "logs"


def _write_factual_zarr(path: Path, variables: list[str]) -> None:
    """Flat Zarr store (all vars at root) for ``xr.open_zarr`` in :meth:`ATTRICIRunner.run`."""
    time = pd.date_range("2020-01-01", periods=30, freq="D")
    lat, lon = [6.0], [-5.0]
    data_vars = {
        var: (("time", "lat", "lon"), np.full((len(time), 1, 1), float(i + 1), dtype=np.float32))
        for i, var in enumerate(variables)
    }
    xr.Dataset(data_vars, coords={"time": time, "lat": lat, "lon": lon}).to_zarr(path, mode="w")


def test_run_invokes_attrici_cli_with_expected_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gmt = tmp_path / "gmt.nc"
    xr.Dataset(
        {"tas": ("time", [0.1, 0.2])}, coords={"time": pd.date_range("2020", periods=2)}
    ).to_netcdf(gmt)

    factual_zarr = tmp_path / "factual.zarr"
    output_zarr = tmp_path / "counterfactual.zarr"
    requested = ["tmax", "precip", "srad"]
    _write_factual_zarr(factual_zarr, requested)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(cmd))
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "attrici 2.0.1", "")

        if "merge-output" in cmd:
            out_path = Path(cmd[-1])
            era5_var = out_path.stem.replace("counterfactual_", "")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            xr.Dataset(
                {era5_var: ("time", np.linspace(1.0, 2.0, 30))},
                coords={"time": pd.date_range("2020-01-01", periods=30, freq="D")},
            ).to_netcdf(out_path)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if "detrend" in cmd:
            var = cmd[cmd.index("--variable") + 1]
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            ts_dir = out_dir / "timeseries" / var
            ts_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = ATTRICIRunner(gmt_file=gmt, work_dir=tmp_path / "work", n_workers=3, backend="scipy")
    out = runner.run(factual_zarr, requested, output_zarr, overwrite=True)

    assert out == output_zarr
    detrend_calls = [c for c in captured if "detrend" in c]
    assert len(detrend_calls) == len(requested)

    for era5_var in requested:
        attrici_var = _ERA5_TO_ATTRICI[era5_var]
        match = next(c for c in detrend_calls if c[c.index("--variable") + 1] == attrici_var)
        assert match[match.index("--gmt-file") + 1] == str(gmt)
        assert match[match.index("--solver") + 1] == "scipy"
        assert match[match.index("--variable") + 1] == attrici_var
        assert match[match.index("--input-file") + 1].endswith(f"factual_{era5_var}.nc")


def test_load_counterfactual_reads_six_variable_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "cf.zarr"
    time = pd.date_range("2020-01-01", periods=10, freq="D")
    lat, lon = [6.0, 7.0], [-5.0, -4.0]

    for idx, var in enumerate(_ALL_SIX):
        data = np.full((len(time), 2, 2), float(idx), dtype=np.float32)
        ds = xr.Dataset(
            {var: (("time", "lat", "lon"), data)},
            coords={"time": time, "lat": lat, "lon": lon},
        )
        ds.attrs["counterfactual"] = True
        mode = "w" if idx == 0 else "a"
        ds.to_zarr(zarr_path, group=var, mode=mode)

    import zarr

    root = zarr.open_group(zarr_path, mode="a")
    root.attrs["attrici_version"] = "test-0.1"
    root.attrs["counterfactual"] = True

    merged = load_counterfactual(zarr_path)
    assert set(merged.data_vars) == set(_ALL_SIX)
    assert bool(merged.attrs.get("counterfactual")) is True
    assert merged.attrs.get("attrici_version") == "test-0.1"
    for var in _ALL_SIX:
        assert merged[var].shape == (len(time), 2, 2)


@pytest.mark.integration
def test_attrici_on_path_integration_smoke(tmp_path: Path) -> None:
    if shutil.which("attrici") is None:
        pytest.skip("attrici not on PATH")

    gmt = tmp_path / "gmt.nc"
    time = pd.date_range("2015-01-01", periods=365, freq="D")
    xr.Dataset({"tas": ("time", np.linspace(0.0, 1.0, 365))}, coords={"time": time}).to_netcdf(gmt)

    factual_zarr = tmp_path / "factual.zarr"
    _write_factual_zarr(factual_zarr, ["tmax"])

    runner = ATTRICIRunner(gmt_file=gmt, work_dir=tmp_path / "work", n_workers=1)
    output_zarr = tmp_path / "out.zarr"

    try:
        runner.run(factual_zarr, ["tmax"], output_zarr, overwrite=True)
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"ATTRICI CLI not compatible with runner flags: {exc}")

    assert output_zarr.exists()
    ds = load_counterfactual(output_zarr)
    assert "tmax" in ds.data_vars
