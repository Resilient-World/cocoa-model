"""Aurora scenario runner: cache keys, mock forecast, optional forward smoke."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_path = _SRC / "counterfactual" / "aurora_runner.py"
_spec = importlib.util.spec_from_file_location("aurora_runner_test", _path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["aurora_runner_test"] = _mod
_spec.loader.exec_module(_mod)

aurora_cache_key = _mod.aurora_cache_key
aurora_cache_path = _mod.aurora_cache_path
load_cached_forecast = _mod.load_cached_forecast
write_cached_forecast = _mod.write_cached_forecast
AuroraScenarioRunner = _mod.AuroraScenarioRunner


def test_aurora_cache_key_construction() -> None:
    key = aurora_cache_key(
        init_time=datetime(2024, 6, 1, 12, 0),
        lead_h=240,
        region="ghana",
        model_size="small",
        lora_id="base",
    )
    assert "ghana" in key
    assert "small" in key
    assert "base" in key
    assert key.endswith("_base")
    assert aurora_cache_path(Path("/tmp/aurora"), key).name == f"{key}.zarr"


def test_aurora_cache_roundtrip(tmp_path: Path) -> None:
    import xarray as xr

    key = aurora_cache_key(
        init_time="20240601T120000",
        lead_h=10,
        region="civ",
        model_size="medium",
        lora_id="civ",
    )
    ds = xr.Dataset({"tmean": ("time", [26.0, 27.0])}, coords={"time": [0, 1]})
    write_cached_forecast(tmp_path, key, ds)
    loaded = load_cached_forecast(tmp_path, key)
    assert loaded is not None
    assert float(loaded["tmean"].isel(time=0)) == 26.0


def test_aurora_mock_forecast_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURORA_MOCK", "1")
    runner = AuroraScenarioRunner(cache_dir=tmp_path, model_size="small", mock=True)
    ds = runner.forecast_farm_point(
        6.5,
        -1.2,
        "ghana",
        ("2050-01-01", "2050-01-15"),
        2050,
    )
    assert "tmean" in ds
    assert ds.sizes["time"] == 15
    assert ds.attrs.get("aurora_backend") == "aurora_mock"


@pytest.mark.slow
@pytest.mark.aurora
def test_aurora_small_forward_smoke() -> None:
    pytest.importorskip("aurora")
    torch = pytest.importorskip("torch")
    from datetime import datetime

    from aurora import AuroraSmallPretrained, Batch, Metadata

    model = AuroraSmallPretrained()
    model.load_checkpoint()
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    pred = model.forward(batch)
    assert "2t" in pred.surf_vars
