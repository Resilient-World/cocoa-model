#!/usr/bin/env python3
"""
Fit per-region ensemble v3 weights via non-negative least squares on held-out tiles.

Blends AEF, Galileo, AgriFM, TerraMind, and FDP → ``config/ensemble_weights_v3.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.cocoa_exposure import REGIONS  # noqa: E402
from data.ensemble_weights import (  # noqa: E402
    DEFAULT_ENSEMBLE_V3_WEIGHTS_PATH,
    V3_BACKEND_KEYS,
    save_ensemble_weights_yaml,
    validate_weights_sum,
)
from scripts.benchmark_backbones import (  # noqa: E402
    DEFAULT_AEF_CKPT,
    DEFAULT_AGRIFM_CKPT,
    DEFAULT_GALILEO_CKPT,
    DEFAULT_TERRAMIND_CKPT,
    AEFHeadPredictor,
    AgriFMPredictor,
    FDPOnlyPredictor,
    GalileoSegPredictor,
    TerraMindPredictor,
    sample_holdout_tiles,
    tile_metrics,
)
from validation.kalischek_benchmark import HeuristicKalischekReference  # noqa: E402

logger = logging.getLogger(__name__)
PREDICTION_THRESHOLD = 0.5


def _predictor_probs(
    predictors: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    from scripts import benchmark_backbones as bb

    out: dict[str, np.ndarray] = {}
    for name, predictor in predictors.items():
        probs = []
        for i, (la, lo) in enumerate(zip(lats, lons, strict=True)):
            batch = bb.build_tile_batch(float(la), float(lo), seed=seed + i)
            probs.append(predictor.predict_tile(batch))
        out[name] = np.stack(probs, axis=0)
    return out


def _tile_mean_probs(prob_maps: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-tile mean probabilities: shape (n_tiles, n_backends)."""
    cols = [prob_maps[k].reshape(prob_maps[k].shape[0], -1).mean(axis=1) for k in V3_BACKEND_KEYS]
    return np.column_stack(cols)


def _tile_mean_labels(labels: np.ndarray) -> np.ndarray:
    return labels.reshape(labels.shape[0], -1).mean(axis=1)


def _blend_maps(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros_like(probs[V3_BACKEND_KEYS[0]], dtype=np.float64)
    for key, w in weights.items():
        blended += w * probs[key]
    return np.clip(blended, 0.0, 1.0)


def _f1_for_blend(labels: np.ndarray, blended: np.ndarray) -> float:
    f1s = []
    for i in range(len(labels)):
        m = tile_metrics(labels[i], blended[i], threshold=PREDICTION_THRESHOLD)
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def fit_nnls_weights(
    prob_maps: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[dict[str, float], float]:
    """NNLS on per-tile mean probabilities; validate F1 on full tile maps."""
    matrix = _tile_mean_probs(prob_maps)
    targets = _tile_mean_labels(labels)
    coef, _ = nnls(matrix, targets)
    if coef.sum() <= 0:
        equal = 1.0 / len(V3_BACKEND_KEYS)
        weights = {k: equal for k in V3_BACKEND_KEYS}
    else:
        weights = {k: float(c) for k, c in zip(V3_BACKEND_KEYS, coef, strict=True)}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    blended = _blend_maps(prob_maps, weights)
    f1 = _f1_for_blend(labels, blended)
    return weights, f1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit ensemble v3 NNLS weights per FDP region")
    parser.add_argument("--n-tiles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="200 tiles/region")
    parser.add_argument("--galileo-checkpoint", type=Path, default=DEFAULT_GALILEO_CKPT)
    parser.add_argument("--aef-checkpoint", type=Path, default=DEFAULT_AEF_CKPT)
    parser.add_argument("--agrifm-checkpoint", type=Path, default=DEFAULT_AGRIFM_CKPT)
    parser.add_argument("--terramind-checkpoint", type=Path, default=DEFAULT_TERRAMIND_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_ENSEMBLE_V3_WEIGHTS_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_tiles = 200 if args.quick else args.n_tiles

    ref = HeuristicKalischekReference()
    predictors = {
        "aef": AEFHeadPredictor(args.aef_checkpoint),
        "galileo": GalileoSegPredictor(args.galileo_checkpoint),
        "agrifm": AgriFMPredictor(args.agrifm_checkpoint),
        "terramind": TerraMindPredictor(args.terramind_checkpoint),
        "fdp": FDPOnlyPredictor(ref),
    }

    doc: dict[str, object] = {
        "schema_version": 1,
        "fitted_date": date.today().isoformat(),
        "default": {},
        "global": {},
        "regions": {},
    }

    all_probs: list[dict[str, np.ndarray]] = []
    all_labels: list[np.ndarray] = []

    for region_key in sorted(REGIONS.keys()):
        logger.info("Fitting ensemble v3 weights for region: %s", region_key)
        lats, lons, labels = sample_holdout_tiles(n_tiles, seed=args.seed, region=region_key)
        probs = _predictor_probs(predictors, lats, lons, seed=args.seed)
        weights, f1 = fit_nnls_weights(probs, labels)
        assert validate_weights_sum(weights)
        doc["regions"][region_key] = {"weights": weights, "f1": round(f1, 4), "n_tiles": len(lats)}
        logger.info("  %s: F1=%.3f weights=%s", region_key, f1, weights)
        all_probs.append(probs)
        all_labels.append(labels)

    merged_probs = {k: np.concatenate([p[k] for p in all_probs], axis=0) for k in V3_BACKEND_KEYS}
    merged_labels = np.concatenate(all_labels, axis=0)
    default_w, default_f1 = fit_nnls_weights(merged_probs, merged_labels)
    doc["default"] = default_w
    doc["global"] = {k: default_w[k] for k in V3_BACKEND_KEYS if k != "fdp"}
    doc["global_f1"] = round(default_f1, 4)

    save_ensemble_weights_yaml(doc, args.out)
    logger.info("Wrote %s (global F1=%.3f)", args.out, default_f1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
