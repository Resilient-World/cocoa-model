"""CPU tests for CorrDiff cache helpers and stratum keys; GPU integration is optional."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from api.online_conformal_store import stratum_key
from counterfactual.corrdiff_downscaler import (
    corrdiff_cache_path,
    load_corrdiff_scenario_ensemble,
    write_synthetic_corrdiff_cache,
)


def test_corrdiff_cache_path_naming(tmp_path: Path) -> None:
    p = corrdiff_cache_path(tmp_path, "ssp245", 2030, "Ghana")
    assert p.name == "corrdiff_ssp245_2030_ghana.zarr"


def test_stratum_key_corrdiff_suffix() -> None:
    assert stratum_key("ssp245", 2030, "ghana") == "ssp245:2030:ghana"
    assert stratum_key("ssp245", 2030, "ghana", downscaling_method="corrdiff") == (
        "ssp245:2030:ghana:corrdiff"
    )


def test_load_corrdiff_scenario_ensemble_synthetic(tmp_path: Path) -> None:
    cache = tmp_path / "corrdiff_ssp245_2030_ghana.zarr"
    write_synthetic_corrdiff_cache(cache, n_samples=3, n_days=365)
    ds = xr.open_zarr(cache)
    assert "sample" in ds.dims
    assert int(ds.sizes["sample"]) == 3

    preset_lat, preset_lon = 6.5, -1.2
    tensors = load_corrdiff_scenario_ensemble(
        cache_path=cache, lat=preset_lat, lon=preset_lon, year=2030
    )
    assert len(tensors) == 3
    assert tensors[0].shape == (1, 365, 11)


def test_corrdiff_cache_missing_message() -> None:
    from counterfactual.corrdiff_downscaler import corrdiff_cache_missing_message

    msg = corrdiff_cache_missing_message(Path("/tmp/x.zarr"), "ssp585", 2080, "ivory_coast")
    assert "run_corrdiff_scenario_bulk" in msg
    assert "ssp585:2080" in msg


@pytest.mark.gpu
def test_corrdiff_import_optional() -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("earth2studio")
