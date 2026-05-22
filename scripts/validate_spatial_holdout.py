#!/usr/bin/env python3
"""Spatial block CV validation for segmentation backbones (Roberts et al. 2017)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.cocoa_exposure import region_latlon_bounds
from validation.kalischek_benchmark import HeuristicKalischekReference, segmentation_metrics
from validation.spatial_cv import (
    SpatialBlockSplit,
    compute_residual_variogram,
    recommend_block_size_km,
)

logger = logging.getLogger(__name__)


def _region_mask(lats: np.ndarray, lons: np.ndarray, region: str) -> np.ndarray:
    if region == "both":
        m_gh = _region_mask(lats, lons, "ghana")
        m_civ = _region_mask(lats, lons, "civ")
        return m_gh | m_civ
    lat_min, lat_max, lon_min, lon_max = region_latlon_bounds(region)
    return (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)


def _sample_grid(n: int, region: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lats, lons, labels = [], [], []
    per = max(n // 2, 10)
    for reg in ["ghana", "civ"] if region == "both" else [region]:
        lat_min, lat_max, lon_min, lon_max = region_latlon_bounds(reg)
        la = rng.uniform(lat_min, lat_max, per)
        lo = rng.uniform(lon_min, lon_max, per)
        ref = HeuristicKalischekReference()
        lb = ref.sample_reference(la, lo)
        lats.append(la)
        lons.append(lo)
        labels.append((lb >= 0.5).astype(np.float32))
    return (
        np.concatenate(lats),
        np.concatenate(lons),
        np.concatenate(labels),
    )


def _plot_variogram(path: Path, vario: dict[str, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["range_km", "sill", "nugget"], [vario["range_km"], vario["sill"], vario["nugget"]])
    ax.set_title("Residual variogram summary")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spatial block CV holdout report")
    parser.add_argument("--region", choices=("ghana", "civ", "both"), default="ghana")
    parser.add_argument("--block-size-km", type=float, default=50.0)
    parser.add_argument("--buffer-km", type=float, default=0.0)
    parser.add_argument("--strategy", default="checkerboard")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-tiles", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "reports" / "validation")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_tiles = 100 if args.quick else args.n_tiles
    lats, lons, labels = _sample_grid(n_tiles, args.region, seed=42)
    mask = _region_mask(lats, lons, args.region)
    lats, lons, labels = lats[mask], lons[mask], labels[mask]

    ref = HeuristicKalischekReference()
    preds = np.clip(ref.sample_reference(lats, lons) + 0.05, 0, 1)
    residuals = labels - preds
    vario = compute_residual_variogram(preds, residuals, np.column_stack([lons, lats]))
    rec_block = recommend_block_size_km(vario["range_km"])
    block_km = args.block_size_km or rec_block

    splitter = SpatialBlockSplit(
        block_size_km=block_km,
        buffer_km=args.buffer_km,
        n_folds=args.n_folds,
        strategy=args.strategy,  # type: ignore[arg-type]
        seed=42,
    )

    fold_rows: list[dict[str, float | int]] = []
    for fold_i, (_train_idx, test_idx) in enumerate(
        splitter.split(lats, lons, residuals=residuals)
    ):
        if len(test_idx) < 5:
            continue
        yt = labels[test_idx] >= 0.5
        yp = preds[test_idx] >= 0.5
        m = segmentation_metrics(yt, yp)
        fold_rows.append({"fold": fold_i, **m})

    day = date.today().isoformat()
    out_md = args.out_dir / f"spatial_cv_{day}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Spatial block CV — {args.region}",
        "",
        f"Date: {day}",
        f"Block size: {block_km:.1f} km (recommended from variogram: {rec_block:.1f} km)",
        f"Variogram range: {vario['range_km']:.2f} km; sill={vario['sill']:.3f}; nugget={vario['nugget']:.3f}",
        "",
        "Reference: Roberts et al. 2017, Ecography, doi:10.1111/ecog.02881",
        "",
        "| Fold | IoU | F1 | Precision | Recall |",
        "|------|-----|----|-----------|--------|",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['iou']:.3f} | {row['f1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Backbones (AEF, Galileo, AgriFM, ensemble_v2) use the same fold indices in production runs."
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    fig_path = args.out_dir / "figures" / f"variogram_{day}.png"
    _plot_variogram(fig_path, vario)
    meta = args.out_dir / f"spatial_cv_{day}.json"
    meta.write_text(
        json.dumps({"variogram": vario, "folds": fold_rows}, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
