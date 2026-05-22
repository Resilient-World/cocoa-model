"""Process-model BMA."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
path = _SRC / "models" / "process" / "bma.py"
spec = importlib.util.spec_from_file_location("bma_mod", path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_bma_mean_and_best() -> None:
    assert mod.combine_predictions(casej=1.0, case2=2.0, almanac=1.5, method="mean") == 1.5
    assert mod.combine_predictions(casej=1.0, case2=2.0, almanac=None, method="best") == 2.0


def test_bma_weighted() -> None:
    y = mod.combine_predictions(casej=1.0, case2=1.0, almanac=1.0, method="bma")
    assert abs(y - 1.0) < 1e-6
