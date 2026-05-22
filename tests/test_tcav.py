"""TCAV analysis smoke test."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_tcav_scores_run() -> None:
    ys_path = _SRC / "models" / "surrogate" / "yield_surrogate.py"
    spec = importlib.util.spec_from_file_location("yield_surrogate_mod", ys_path)
    assert spec and spec.loader
    ys_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ys_mod)

    tcav_path = _SRC / "analysis" / "tcav.py"
    spec2 = importlib.util.spec_from_file_location("analysis.tcav", tcav_path)
    assert spec2 and spec2.loader
    tcav_mod = importlib.util.module_from_spec(spec2)
    sys.modules["analysis.tcav"] = tcav_mod
    spec2.loader.exec_module(tcav_mod)

    model = ys_mod.YieldSurrogateModel()
    climate = torch.randn(6, 365, 11)
    static = torch.randn(6, 13)
    results = tcav_mod.tcav_scores(model, climate=climate, static=static, n_random=10)
    assert len(results) >= 1
    assert 0.0 <= results[0].score <= 1.0
