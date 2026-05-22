#!/usr/bin/env python3
"""
Grid-search per-region ensemble v2 weights on held-out Kalischek tiles.

Blends AEF, Galileo, AgriFM, and FDP to maximize F1 vs in-situ reference labels.
Writes ``config/ensemble_weights.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.cocoa_exposure import REGIONS
from data.ensemble_weights import (
    BACKEND_KEYS,
    DEFAULT_ENSEMBLE_WEIGHTS_PATH,
    save_ensemble_weights_yaml,
    validate_weights_sum,
)
from scripts.benchmark_backbones import (
    DEFAULT_AEF_CKPT,
    DEFAULT_AGRIFM_CKPT,
    DEFAULT_GALILEO_CKPT,
    AEFHeadPredictor,
    AgriFMPredictor,
    FDPOnlyPredictor,
    GalileoSegPredictor,
    sample_holdout_tiles,
    tile_metrics,
)

logger = logging.getLogger(__name__)
PREDICTION_THRESHOLD = 0.5
GRID_STEP = 0.05


def _grid_weights(step: float = GRID_STEP) -> list[dict[str, float]]:
    """Coarse 4-way simplex grid (aef, galileo, agrifm, fdp)."""
    combos: list[dict[str, float]] = []
    steps = np.arange(0.0, 1.0 + step / 2, step)
    for w_aef in steps:
        for w_gal in steps:
            for w_ag in steps:
                w_fdp = 1.0 - w_aef - w_gal - w_ag
                if w_fdp < -1e-9 or w_fdp > 1.0 + 1e-9:
                    continue
                if abs(w_aef + w_gal + w_ag + w_fdp - 1.0) > 1e-6:
                    continue
                combos.append(
                    {
                        "aef": float(w_aef),
                        "galileo": float(w_gal),
                        "agrifm": float(w_ag),
                        "fdp": float(max(0.0, w_fdp)),
                    }
                )
    return combos


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


def _blend(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros_like(next(iter(probs.values())), dtype=np.float64)
    for key, w in weights.items():
        if key in probs:
            blended += w * probs[key]
    return np.clip(blended, 0.0, 1.0)


def _f1_for_blend(labels: np.ndarray, prob_maps: np.ndarray, blended: np.ndarray) -> float:
    f1s = []
    for i in range(len(labels)):
        m = tile_metrics(labels[i], blended[i], threshold=PREDICTION_THRESHOLD)
        f1s.append(m["f1"])
    return float(np.mean(f1s))


def fit_region(
    region_key: str,
    *,
    n_tiles: int,
    seed: int,
    galileo_ckpt: Path,
    aef_ckpt: Path,
    agrifm_ckpt: Path,
    grid_step: float,
) -> dict[str, object]:
    lats, lons, labels = sample_holdout_tiles(n_tiles, seed=seed, region=region_key)
    from validation.kalischek_benchmark import HeuristicKalischekReference

    ref = HeuristicKalischekReference()
    predictors = {
        "aef": AEFHeadPredictor(aef_ckpt),
        "galileo": GalileoSegPredictor(galileo_ckpt),
        "agrifm": AgriFMPredictor(agrifm_ckpt),
        "fdp": FDPOnlyPredictor(ref),
    }
    probs = _predictor_probs(predictors, lats, lons, seed=seed)

    best_w: dict[str, float] | None = None
    best_f1 = -1.0
    single_f1: dict[str, float] = {}
    for key in BACKEND_KEYS:
        single_f1[key] = _f1_for_blend(labels, probs, probs[key])

    for weights in _grid_weights(grid_step):
        blended = _blend(probs, weights)
        f1 = _f1_for_blend(labels, probs, blended)
        if f1 > best_f1:
            best_f1 = f1
            best_w = weights

    if best_w is None:
        best_w = {"aef": 0.4, "galileo": 0.25, "agrifm": 0.25, "fdp": 0.10}
        best_f1 = _f1_for_blend(labels, probs, _blend(probs, best_w))

    return {
        "weights": best_w,
        "val_f1": best_f1,
        "val_f1_best_single": single_f1,
        "n_tiles": len(lats),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit ensemble v2 weights per FDP region")
    parser.add_argument("--n-tiles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP)
    parser.add_argument("--quick", action="store_true", help="200 tiles/region, coarser grid")
    parser.add_argument("--galileo-checkpoint", type=Path, default=DEFAULT_GALILEO_CKPT)
    parser.add_argument("--aef-checkpoint", type=Path, default=DEFAULT_AEF_CKPT)
    parser.add_argument("--agrifm-checkpoint", type=Path, default=DEFAULT_AGRIFM_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_ENSEMBLE_WEIGHTS_PATH)
    parser.add_argument("--min-regions-pass", type=int, default=6)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_tiles = 200 if args.quick else args.n_tiles
    grid_step = 0.10 if args.quick else args.grid_step

    doc: dict[str, object] = {
        "schema_version": 1,
        "fitted_date": date.today().isoformat(),
        "default": {"aef": 0.40, "galileo": 0.25, "agrifm": 0.25, "fdp": 0.10},
        "global": {"aef": 0.45, "galileo": 0.30, "agrifm": 0.25},
        "regions": {},
    }

    regions_pass = 0
    for key in sorted(REGIONS.keys()):
        logger.info("Fitting ensemble weights for region: %s", key)
        result = fit_region(
            key,
            n_tiles=n_tiles,
            seed=args.seed,
            galileo_ckpt=args.galileo_checkpoint,
            aef_ckpt=args.aef_checkpoint,
            agrifm_ckpt=args.agrifm_checkpoint,
            grid_step=grid_step,
        )
        weights = result["weights"]
        assert validate_weights_sum(weights)
        best_single = max(result["val_f1_best_single"].values())
        if result["val_f1"] >= best_single - 1e-6:
            regions_pass += 1
        doc["regions"][key] = result
        logger.info(
            "  %s: ensemble F1=%.3f best_single=%.3f weights=%s",
            key,
            result["val_f1"],
            best_single,
            weights,
        )

    save_ensemble_weights_yaml(doc, args.out)
    logger.info("Wrote %s (%d/%d regions pass F1 gate)", args.out, regions_pass, len(REGIONS))
    if regions_pass < args.min_regions_pass:
        logger.warning(
            "Acceptance gate: ensemble F1 >= best single on only %d/%d regions (need %d)",
            regions_pass,
            len(REGIONS),
            args.min_regions_pass,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
