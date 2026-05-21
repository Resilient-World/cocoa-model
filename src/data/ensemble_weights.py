"""
Per-region ensemble weights for exposure backend ``ensemble_v2``.

Weights are fit via :mod:`scripts.fit_ensemble_v2_weights` on held-out Kalischek tiles
and stored in ``config/ensemble_weights.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from data.cocoa_exposure import normalize_region_key

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENSEMBLE_WEIGHTS_PATH = _REPO_ROOT / "config" / "ensemble_weights.yaml"

BACKEND_KEYS = ("aef", "galileo", "agrifm", "fdp")
GLOBAL_BACKEND_KEYS = ("aef", "galileo", "agrifm")


def _normalize_weights(weights: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    subset = {k: float(weights.get(k, 0.0)) for k in keys}
    total = sum(subset.values())
    if total <= 0:
        equal = 1.0 / len(keys)
        return {k: equal for k in keys}
    return {k: v / total for k, v in subset.items()}


def validate_weights_sum(weights: dict[str, float], *, tol: float = 1e-3) -> bool:
    total = sum(weights.values())
    return abs(total - 1.0) <= tol


def load_ensemble_weights_yaml(path: Path | str = DEFAULT_ENSEMBLE_WEIGHTS_PATH) -> dict[str, Any]:
    """Load raw YAML document."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Ensemble weights file missing at %s; using built-in defaults", path)
        return _builtin_defaults()
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _builtin_defaults() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "default": {"aef": 0.40, "galileo": 0.25, "agrifm": 0.25, "fdp": 0.10},
        "global": {"aef": 0.45, "galileo": 0.30, "agrifm": 0.25},
        "regions": {},
    }


def load_ensemble_weights(
    region: str | None = None,
    *,
    path: Path | str = DEFAULT_ENSEMBLE_WEIGHTS_PATH,
    global_fallback: bool = False,
) -> dict[str, float]:
    """
    Return normalized weights for ``ensemble_v2``.

    Parameters
    ----------
    region:
        FDP region key (``ghana``, ``civ``, …). Uses ``default`` when unknown.
    global_fallback:
        If True, return ``global`` block (no FDP) for points outside native coverage.
    """
    doc = load_ensemble_weights_yaml(path)
    if global_fallback:
        block = doc.get("global") or doc.get("default", {})
        return _normalize_weights(block, GLOBAL_BACKEND_KEYS)

    region_key = normalize_region_key(region) if region else None
    regions = doc.get("regions") or {}
    if region_key and region_key in regions:
        entry = regions[region_key]
        block = entry.get("weights", entry)
    else:
        block = doc.get("default", {})
    return _normalize_weights(block, BACKEND_KEYS)


def save_ensemble_weights_yaml(doc: dict[str, Any], path: Path | str = DEFAULT_ENSEMBLE_WEIGHTS_PATH) -> Path:
    """Persist ensemble weights document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, default_flow_style=False, sort_keys=False)
    return path
