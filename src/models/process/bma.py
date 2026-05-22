"""Bayesian model averaging over process-model yield predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

ProcessEnsembleMethod = Literal["mean", "bma", "best"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "config" / "process_bma_weights.json"


def load_bma_weights(path: Path | str = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    p = Path(path)
    if not p.is_file():
        return {"casej": 0.5, "case2": 0.25, "almanac": 0.25}
    data = json.loads(p.read_text(encoding="utf-8"))
    w = {k: float(data.get(k, 0.0)) for k in ("casej", "case2", "almanac")}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def combine_predictions(
    *,
    casej: float | None,
    case2: float | None,
    almanac: float | None,
    method: ProcessEnsembleMethod = "mean",
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
) -> float:
    """Combine available process model yields (tonnes/ha)."""
    parts: dict[str, float] = {}
    if casej is not None:
        parts["casej"] = casej
    if case2 is not None:
        parts["case2"] = case2
    if almanac is not None:
        parts["almanac"] = almanac
    if not parts:
        raise ValueError("No process model predictions available for BMA")
    if method == "best":
        return max(parts.values())
    if method == "mean":
        return float(np.mean(list(parts.values())))
    weights = load_bma_weights(weights_path)
    score = sum(weights.get(k, 0.0) * v for k, v in parts.items())
    wsum = sum(weights.get(k, 0.0) for k in parts)
    return float(score / max(wsum, 1e-9))
