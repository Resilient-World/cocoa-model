#!/usr/bin/env python3
"""Fit ensemble v4 NNLS weights; gate on OlmoEarth-Base vs v3 F1 (+2pp on >=4 regions)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.ensemble_weights import (  # noqa: E402
    DEFAULT_ENSEMBLE_V4_WEIGHTS_PATH,
    V4_BACKEND_KEYS,
    save_ensemble_weights_yaml,
    validate_weights_sum,
)
from scripts import benchmark_backbones as bb  # noqa: E402
from scripts.fit_ensemble_v3_weights import (  # noqa: E402
    _blend_maps,
    _f1_for_blend,
    _predictor_probs,
    _tile_mean_labels,
)


def _tile_mean_probs_v4(prob_maps: dict[str, np.ndarray]) -> np.ndarray:
    cols = [prob_maps[k].reshape(prob_maps[k].shape[0], -1).mean(axis=1) for k in V4_BACKEND_KEYS]
    return np.column_stack(cols)


def _blend_maps_v4(prob_maps: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros_like(prob_maps[V4_BACKEND_KEYS[0]], dtype=np.float64)
    for key, w in weights.items():
        blended += w * prob_maps[key]
    return np.clip(blended, 0.0, 1.0)
from validation.kalischek_benchmark import HeuristicKalischekReference  # noqa: E402

logger = logging.getLogger(__name__)
PROMOTION_REGIONS = bb.BENCHMARK_REGIONS_SIX
PROMOTION_MARGIN = 0.02
PROMOTION_MIN_REGIONS = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_ENSEMBLE_V4_WEIGHTS_PATH)
    parser.add_argument("--skip-promotion-check", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.synthetic:
        from data.ensemble_weights import _builtin_v4_defaults

        doc = _builtin_v4_defaults()
        doc["fitted_date"] = date.today().isoformat()
        save_ensemble_weights_yaml(doc, args.out)
        return 0

    ref = HeuristicKalischekReference()
    predictors = {
        "olmoearth": bb.OlmoEarthPredictor(bb._olmoearth_ckpt("base"), model_size="base"),
        "agrifm": bb.AgriFMPredictor(bb.DEFAULT_AGRIFM_CKPT),
        "terramind": bb.TerraMindPredictor(bb.DEFAULT_TERRAMIND_CKPT),
        "galileo": bb.GalileoSegPredictor(bb.DEFAULT_GALILEO_CKPT),
        "aef": bb.AEFHeadPredictor(bb.DEFAULT_AEF_CKPT),
        "fdp": bb.FDPOnlyPredictor(ref),
    }
    v3 = None
    regions_pass = 0
    doc: dict[str, object] = {"schema_version": 1, "fitted_date": date.today().isoformat(), "regions": {}}

    for region_key in PROMOTION_REGIONS:
        lats, lons, labels = bb.sample_holdout_tiles(args.n_tiles, seed=args.seed, region=region_key)
        probs = _predictor_probs(predictors, lats, lons, seed=args.seed)
        matrix = _tile_mean_probs_v4(probs)
        targets = _tile_mean_labels(labels)
        coef, _ = nnls(matrix, targets)
        total = float(coef.sum()) or 1.0
        weights = {k: float(c) / total for k, c in zip(V4_BACKEND_KEYS, coef, strict=True)}
        blended = _blend_maps_v4(probs, weights)
        f1_v4 = _f1_for_blend(labels, blended)
        if v3 is None:
            v3 = bb.EnsembleV3Predictor(
                region=region_key,
                galileo_checkpoint=bb.DEFAULT_GALILEO_CKPT,
                aef_checkpoint=bb.DEFAULT_AEF_CKPT,
                agrifm_checkpoint=bb.DEFAULT_AGRIFM_CKPT,
                terramind_checkpoint=bb.DEFAULT_TERRAMIND_CKPT,
            )
        v3_res = bb.evaluate_predictor(v3, lats, lons, labels, max_latency_tiles=10)
        oe_only = bb.evaluate_predictor(predictors["olmoearth"], lats, lons, labels, max_latency_tiles=10)
        if oe_only.f1 - v3_res.f1 > PROMOTION_MARGIN:
            regions_pass += 1
        doc["regions"][region_key] = {"weights": weights, "f1": round(f1_v4, 4), "f1_v3": round(v3_res.f1, 4)}
        logger.info("%s: v4 F1=%.3f v3=%.3f olmoearth=%.3f", region_key, f1_v4, v3_res.f1, oe_only.f1)

    doc["promotion"] = {"regions_pass": regions_pass, "required": PROMOTION_MIN_REGIONS}
    save_ensemble_weights_yaml(doc, args.out)

    if not args.skip_promotion_check and regions_pass < PROMOTION_MIN_REGIONS:
        logger.error(
            "Promotion gate failed: OlmoEarth-Base > v3+2pp in %s/%s regions",
            regions_pass,
            PROMOTION_MIN_REGIONS,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
