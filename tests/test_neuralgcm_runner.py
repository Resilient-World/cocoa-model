"""NeuralGCM runner stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
path = _SRC / "counterfactual" / "neuralgcm_runner.py"
spec = importlib.util.spec_from_file_location("neuralgcm_runner", path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["neuralgcm_runner"] = mod
spec.loader.exec_module(mod)


def test_emulate_returns_dataset() -> None:
    ds = mod.emulate_era5_point(lat=6.0, lon=-2.0, start="2020-01-01", end="2020-01-15")
    assert "tmean" in ds
    assert ds.sizes["time"] == 15
