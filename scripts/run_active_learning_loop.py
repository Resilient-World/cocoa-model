#!/usr/bin/env python3
"""Run BSSAL active learning loops for sparse-label cocoa expansion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from data.cocoa_exposure import normalize_region_key, region_latlon_bounds
from training.active_learner import BSSALCocoaLearner
from training.ssl_pseudo import make_pseudo_labels, should_run_self_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_geojson_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    doc = json.loads(path.read_text())
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    features: list[list[float]] = []
    for item in doc.get("features", []):
        geom = item.get("geometry") or {}
        props = item.get("properties") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom.get("coordinates", [None, None])[:2]
        label = props.get("label", props.get("cocoa_label", props.get("cocoa")))
        if lon is None or lat is None or label is None:
            continue
        coords.append((float(lon), float(lat)))
        labels.append(int(label))
        features.append(_feature_vector(float(lon), float(lat)))
    return (
        np.asarray(coords, dtype=np.float64),
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _write_geojson(path: Path, lonlat: np.ndarray, props: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for (lon, lat), item_props in zip(lonlat, props, strict=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": item_props,
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))


def _feature_vector(lon: float, lat: float) -> list[float]:
    return [
        lon,
        lat,
        np.sin(np.deg2rad(lat)),
        np.cos(np.deg2rad(lon)),
        float(-85.0 <= lon <= -66.0 and -15.0 <= lat <= 12.0),
    ]


def _synthetic_initial_labels(region: str, path: Path, *, n: int = 120, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    lat_min, lat_max, lon_min, lon_max = region_latlon_bounds(region)
    lon = rng.uniform(lon_min, lon_max, n)
    lat = rng.uniform(lat_min, lat_max, n)
    score = (
        np.exp(-((lat - (lat_min + lat_max) / 2.0) ** 2) / 18.0)
        + 0.35 * np.sin(np.deg2rad(lon * 4.0))
        + rng.normal(0.0, 0.08, n)
    )
    labels = (score > np.median(score)).astype(int)
    _write_geojson(
        path,
        np.column_stack([lon, lat]),
        [{"label": int(label), "source": "synthetic_ci"} for label in labels],
    )
    return path


def _candidate_pool(
    region: str,
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat_min, lat_max, lon_min, lon_max = region_latlon_bounds(region)
    lon = rng.uniform(lon_min, lon_max, n)
    lat = rng.uniform(lat_min, lat_max, n)
    lonlat = np.column_stack([lon, lat]).astype(np.float64)
    X = np.asarray([_feature_vector(float(lo), float(la)) for lo, la in lonlat], dtype=np.float64)
    month = np.arange(12, dtype=np.float64)
    seasonal = np.sin((month[None, :] + lat[:, None]) / 12.0 * 2.0 * np.pi)
    gradient = (lon[:, None] - lon_min) / max(lon_max - lon_min, 1e-6)
    ndvi = 0.55 + 0.12 * seasonal + 0.05 * gradient + rng.normal(0.0, 0.015, (n, 12))
    return lonlat, X, ndvi.astype(np.float64)


def _append_pseudo_labels(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_candidates: np.ndarray,
    learner: BSSALCocoaLearner,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    probabilities = learner.predict_proba(X_candidates)
    if probabilities.shape[1] == 2:
        cocoa_prob = probabilities[:, 1:2]
    else:
        cocoa_prob = probabilities.max(axis=1, keepdims=True)
    import torch

    pseudo = make_pseudo_labels(torch.from_numpy(cocoa_prob.astype(np.float32)), threshold=threshold)
    mask = pseudo.mask.squeeze(-1).numpy().astype(bool)
    if not np.any(mask):
        return X_labeled, y_labeled, 0
    y_pseudo = pseudo.pseudo_labels.squeeze(-1).numpy().astype(np.int64)[mask]
    X_aug = np.vstack([X_labeled, X_candidates[mask]])
    y_aug = np.concatenate([y_labeled, y_pseudo])
    return X_aug, y_aug, int(mask.sum())


def _write_report(
    path: Path,
    *,
    region: str,
    iteration: int,
    selected: int,
    range_m: float,
    pseudo_count: int,
    to_label_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# Active learning report — {region} iter {iteration}",
                "",
                f"- Selected for review: {selected}",
                f"- Spatial uncorrelation range (m): {range_m:.2f}",
                f"- High-confidence pseudo-labels added: {pseudo_count}",
                f"- Human label packet: `{to_label_path}`",
                "- Next step: label GeoJSON in QGIS / FieldNotes and feed back as initial labels.",
                "",
            ]
        )
    )


def run_loop(args: argparse.Namespace) -> int:
    region = normalize_region_key(args.region)
    initial_labels = args.initial_labels
    if initial_labels is None:
        initial_labels = REPO_ROOT / "data" / "active" / region / "synthetic_initial_labels.geojson"
        _synthetic_initial_labels(region, initial_labels)

    labeled_lonlat, X_labeled, y_labeled = _load_geojson_labels(initial_labels)
    report_root = REPO_ROOT / "reports" / "active" / region
    active_root = REPO_ROOT / "data" / "active" / region

    try:
        import mlflow

        mlflow.set_experiment(f"active_learning_{region}")
    except Exception:
        mlflow = None

    learner = BSSALCocoaLearner(random_state=42)
    for iteration in range(1, args.iterations + 1):
        learner.fit(X_labeled, y_labeled)
        candidate_lonlat, X_candidates, monthly_ndvi = _candidate_pool(
            region,
            n=max(args.budget * 20, 500),
            seed=1000 + iteration,
        )
        query = learner.query(
            X_candidates,
            candidate_lonlat,
            labeled_lonlat,
            budget=args.budget,
            monthly_ndvi=monthly_ndvi,
        )
        selected_lonlat = candidate_lonlat[query.indices]
        props = [
            {
                "iteration": iteration,
                "rank": rank + 1,
                "vote_entropy": float(entropy),
                "needs_human_label": True,
            }
            for rank, entropy in enumerate(query.entropy)
        ]
        to_label = active_root / f"iter_{iteration}" / "to_label.geojson"
        _write_geojson(to_label, selected_lonlat, props)

        pseudo_count = 0
        if should_run_self_training(iteration, every_k=args.self_train_every):
            X_labeled, y_labeled, pseudo_count = _append_pseudo_labels(
                X_labeled,
                y_labeled,
                X_candidates,
                learner,
                threshold=args.pseudo_threshold,
            )

        report = report_root / f"iter_{iteration}.md"
        _write_report(
            report,
            region=region,
            iteration=iteration,
            selected=len(query.indices),
            range_m=query.range_m,
            pseudo_count=pseudo_count,
            to_label_path=to_label,
        )

        if mlflow is not None:
            with mlflow.start_run(run_name=f"{region}_iter_{iteration}", nested=True):
                mlflow.log_param("region", region)
                mlflow.log_param("budget", args.budget)
                mlflow.log_metric("selected", len(query.indices))
                mlflow.log_metric("spatial_range_m", query.range_m)
                mlflow.log_metric("pseudo_labels", pseudo_count)
                mlflow.log_artifact(str(report))

        if len(selected_lonlat):
            new_y = learner.predict_proba(X_candidates[query.indices])[:, -1] >= 0.5
            labeled_lonlat = np.vstack([labeled_lonlat, selected_lonlat])
            X_labeled = np.vstack([X_labeled, X_candidates[query.indices]])
            y_labeled = np.concatenate([y_labeled, new_y.astype(np.int64)])

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BSSAL cocoa active learning loop")
    parser.add_argument("--region", required=True, help="Region code/name, e.g. peru")
    parser.add_argument("--initial-labels", type=Path, default=None)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--pseudo-threshold", type=float, default=float(os.getenv("PSEUDO_THRESHOLD", 0.95)))
    parser.add_argument("--self-train-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_loop(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
