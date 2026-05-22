#!/usr/bin/env python3
"""
Calibrate FDP 2025a probability threshold against Kalischek et al. (2023) in-situ labels.

Samples 5,000 stratified points over Côte d'Ivoire + Ghana, sweeps thresholds
[0.50, 0.99] step 0.01, and writes ``reports/fdp_calibration_<date>.md`` plus a
precision–recall curve (matplotlib).

Requires Earth Engine credentials unless ``--mock`` is passed (synthetic CI run).

Example::

    python scripts/calibrate_fdp_threshold.py
    python scripts/calibrate_fdp_threshold.py --mock --n-points 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.cocoa_exposure import (
    DEFAULT_THRESHOLD,
    FDP_COCOA_COLLECTION,
    FDP_MODEL_CARD_URL,
    PROBABILITY_BAND,
)
from validation.kalischek_benchmark import (
    DEFAULT_KALISCHEK_ASSET,
    REGIONS,
)

logger = logging.getLogger(__name__)
DEFAULT_REPORT_DIR = _REPO_ROOT / "reports"
FDP_YEAR = 2023
REFERENCE_THRESHOLD = 0.5
SWEEP_START = 0.50
SWEEP_STOP = 0.99
SWEEP_STEP = 0.01


def threshold_sweep_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Precision, recall, F1, and Youden's J for each threshold."""
    yt = y_true.astype(bool)
    rows: list[dict[str, float]] = []
    n_pos = float(yt.sum())
    n_neg = float((~yt).sum())

    for t in thresholds:
        yp = y_prob >= t
        tp = float(np.sum(yt & yp))
        fp = float(np.sum(~yt & yp))
        fn = float(np.sum(yt & ~yp))
        tn = float(np.sum(~yt & ~yp))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / n_neg if n_neg > 0 else 0.0
        sensitivity = recall
        youden_j = sensitivity + specificity - 1.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "youden_j": youden_j,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def _sample_grid(region: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = REGIONS[region]
    rng = np.random.default_rng(seed)
    lats = rng.uniform(lat_min, lat_max, n)
    lons = rng.uniform(lon_min, lon_max, n)
    return lats.astype(np.float64), lons.astype(np.float64)


def stratified_subsample(
    lats: np.ndarray,
    lons: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balance cocoa / non-cocoa strata from Kalischek labels."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true.astype(bool))[0]
    neg_idx = np.where(~y_true.astype(bool))[0]
    half = n_points // 2
    n_pos = min(half, len(pos_idx))
    n_neg = min(n_points - n_pos, len(neg_idx))
    if n_pos < half and len(neg_idx) >= n_points - n_pos:
        n_neg = n_points - n_pos
    elif n_neg < half and len(pos_idx) >= n_points - n_neg:
        n_pos = n_points - n_neg

    pick_pos = rng.choice(pos_idx, size=n_pos, replace=len(pos_idx) < n_pos)
    pick_neg = rng.choice(neg_idx, size=n_neg, replace=len(neg_idx) < n_neg)
    idx = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(idx)
    return lats[idx], lons[idx], y_true[idx], y_prob[idx]


def synthetic_calibration_data(
    n_points: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced mock data with F1 peak near 0.96 for offline runs."""
    rng = np.random.default_rng(seed)
    lats, lons = [], []
    for i, region in enumerate(REGIONS):
        la, lo = _sample_grid(region, n_points // 2 + 100, seed=seed + i)
        lats.append(la)
        lons.append(lo)
    lat_all = np.concatenate(lats)
    lon_all = np.concatenate(lons)

    n_pool = len(lat_all)
    y_true = rng.random(n_pool) < 0.5
    # Cocoa pixels high probability; non-cocoa low — F1 peak near 0.96 (FDP model card)
    y_prob = np.where(
        y_true,
        rng.beta(40, 2, n_pool) * 0.08 + 0.90,
        rng.beta(2, 40, n_pool) * 0.35,
    )
    return stratified_subsample(lat_all, lon_all, y_true, y_prob, n_points=n_points, seed=seed)


def load_gee_samples(
    *,
    n_points: int,
    seed: int,
    kalischek_asset: str,
    project: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample FDP 2025a probabilities and Kalischek reference over CIV + GHA."""
    import ee

    from data.gee_auth import initialize_earth_engine

    initialize_earth_engine(project=project)

    pool_n = max(n_points * 3, 6000)
    lats_list, lons_list = [], []
    for i, region in enumerate(REGIONS):
        la, lo = _sample_grid(region, pool_n // 2, seed=seed + i)
        lats_list.append(la)
        lons_list.append(lo)
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)

    kal_image = ee.Image(kalischek_asset)
    kal_band = kal_image.bandNames().getInfo()[0]

    fdp = (
        ee.ImageCollection(FDP_COCOA_COLLECTION)
        .filterDate("2023-01-01", "2023-12-31")
        .mosaic()
        .select(PROBABILITY_BAND)
    )

    kal_probs: list[float] = []
    fdp_probs: list[float] = []
    batch = 400
    for start in range(0, len(lats), batch):
        sl = slice(start, min(start + batch, len(lats)))
        features = [
            ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"id": int(i)})
            for i, (la, lo) in enumerate(zip(lats[sl], lons[sl], strict=True))
        ]
        fc = ee.FeatureCollection(features)
        kal_sampled = kal_image.select(kal_band).reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=10,
        )
        fdp_sampled = fdp.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=10,
        )
        for feat in kal_sampled.getInfo()["features"]:
            raw = feat["properties"].get(kal_band, 0.0) or 0.0
            val = float(raw)
            kal_probs.append(val / 100.0 if val > 1.0 else val)
        for feat in fdp_sampled.getInfo()["features"]:
            raw = feat["properties"].get(PROBABILITY_BAND, 0.0) or 0.0
            fdp_probs.append(float(np.clip(raw, 0.0, 1.0)))

    kal_arr = np.clip(np.array(kal_probs, dtype=np.float64), 0.0, 1.0)
    fdp_arr = np.clip(np.array(fdp_probs, dtype=np.float64), 0.0, 1.0)
    y_true = kal_arr >= REFERENCE_THRESHOLD

    valid = np.isfinite(fdp_arr) & np.isfinite(kal_arr)
    return stratified_subsample(
        lats[valid],
        lons[valid],
        y_true[valid],
        fdp_arr[valid],
        n_points=n_points,
        seed=seed,
    )


def write_report(
    metrics: pd.DataFrame,
    path: Path,
    *,
    n_points: int,
    mock: bool,
    pr_figure: Path | None,
) -> None:
    best_f1 = metrics.loc[metrics["f1"].idxmax()]
    best_youden = metrics.loc[metrics["youden_j"].idxmax()]
    lines = [
        f"# FDP 2025a threshold calibration ({date.today().isoformat()})",
        "",
        f"Stratified **{n_points}** points over Côte d'Ivoire + Ghana vs Kalischek et al. "
        f"(2023) in-situ reference (`{DEFAULT_KALISCHEK_ASSET}`).",
        "",
        f"- FDP collection: `{FDP_COCOA_COLLECTION}` ({FDP_YEAR})",
        f"- Model card (F1-optimal ≈ **0.96**): {FDP_MODEL_CARD_URL}",
        f"- Repo default threshold: **{DEFAULT_THRESHOLD}**",
        f"- Data mode: **{'synthetic (mock)' if mock else 'Earth Engine'}**",
        "",
        "## Best thresholds",
        "",
        "| Criterion | Threshold | Precision | Recall | F1 | Youden J |",
        "|-----------|-----------|-----------|--------|-----|----------|",
        f"| Max F1 | {best_f1['threshold']:.2f} | {best_f1['precision']:.3f} | "
        f"{best_f1['recall']:.3f} | {best_f1['f1']:.3f} | {best_f1['youden_j']:.3f} |",
        f"| Max Youden J | {best_youden['threshold']:.2f} | {best_youden['precision']:.3f} | "
        f"{best_youden['recall']:.3f} | {best_youden['f1']:.3f} | {best_youden['youden_j']:.3f} |",
        "",
    ]
    if pr_figure is not None:
        lines.extend(
            [
                f"![Precision–recall curve]({pr_figure.name})",
                "",
            ]
        )
    lines.append("## Sweep (selected thresholds)")
    lines.append("")
    lines.append("| Threshold | Precision | Recall | F1 | Youden J |")
    lines.append("|-----------|-----------|--------|-----|----------|")
    for _, row in metrics.iloc[::5].iterrows():
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['youden_j']:.3f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pr_curve(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(metrics["recall"], metrics["precision"], marker=".", markersize=3, label="FDP sweep")
    best = metrics.loc[metrics["f1"].idxmax()]
    ax.scatter(
        [best["recall"]],
        [best["precision"]],
        color="crimson",
        zorder=5,
        label=f"max F1 @ {best['threshold']:.2f}",
    )
    ref = metrics.loc[(metrics["threshold"] - DEFAULT_THRESHOLD).abs().idxmin()]
    ax.scatter(
        [ref["recall"]],
        [ref["precision"]],
        color="green",
        zorder=5,
        label=f"default {DEFAULT_THRESHOLD}",
    )
    ax.set_xlabel("Recall (Kalischek in-situ)")
    ax.set_ylabel("Precision")
    ax.set_title("FDP 2025a precision–recall vs threshold")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_calibration(
    *,
    n_points: int = 5000,
    seed: int = 42,
    mock: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
    kalischek_asset: str = DEFAULT_KALISCHEK_ASSET,
    project: str | None = None,
) -> Path:
    thresholds = np.arange(SWEEP_START, SWEEP_STOP + SWEEP_STEP / 2, SWEEP_STEP)

    if mock:
        _lats, _lons, y_true, y_prob = synthetic_calibration_data(n_points, seed=seed)
    else:
        _lats, _lons, y_true, y_prob = load_gee_samples(
            n_points=n_points,
            seed=seed,
            kalischek_asset=kalischek_asset,
            project=project,
        )

    metrics = threshold_sweep_metrics(y_true, y_prob, thresholds)
    stamp = date.today().isoformat()
    report_path = report_dir / f"fdp_calibration_{stamp}.md"
    pr_path = report_dir / f"fdp_calibration_{stamp}_pr.png"
    plot_pr_curve(metrics, pr_path)
    write_report(metrics, report_path, n_points=n_points, mock=mock, pr_figure=pr_path)
    best_t = float(metrics.loc[metrics["f1"].idxmax(), "threshold"])
    logger.info("Wrote %s and %s (max F1 threshold=%.2f)", report_path, pr_path, best_t)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate FDP cocoa probability threshold")
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Synthetic data (no GEE)")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kalischek-asset", type=str, default=DEFAULT_KALISCHEK_ASSET)
    parser.add_argument("--project", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_calibration(
        n_points=args.n_points,
        seed=args.seed,
        mock=args.mock,
        report_dir=args.report_dir,
        kalischek_asset=args.kalischek_asset,
        project=args.project,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
