#!/usr/bin/env python3
"""
Train :class:`models.aef_cocoa_head.AEFCocoaHead` on AlphaEarth embeddings + Kalischek labels.

Samples stratified points over Côte d'Ivoire and Ghana (GEE Kalischek asset when
credentials exist; belt heuristic otherwise), fits a 64→128→1 MLP, and saves
``models/aef_cocoa_head.pt``.

Example::

    python scripts/train_aef_head.py --epochs 30
    python scripts/train_aef_head.py --synthetic --epochs 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.alphaearth_embeddings import AEF_BAND_NAMES, AEF_EMBEDDING_DIM
from models.aef_cocoa_head import AEFCocoaHead, DEFAULT_AEF_CHECKPOINT
from validation.kalischek_benchmark import (
    DEFAULT_KALISCHEK_ASSET,
    GeeKalischekReference,
    HeuristicKalischekReference,
    REGIONS,
    _sample_grid,
)

logger = logging.getLogger(__name__)


def _synthetic_embeddings_and_labels(
    n: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic unit-norm embeddings correlated with cocoa labels (offline CI)."""
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.45
    base = rng.normal(0, 1, (n, AEF_EMBEDDING_DIM)).astype(np.float32)
    # Shift positive class along a learned-like direction
    direction = rng.normal(0, 1, AEF_EMBEDDING_DIM).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    base[labels] += 0.35 * direction
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    embeddings = base / np.clip(norms, 1e-6, None)
    y = labels.astype(np.float32)
    return embeddings, y


def _sample_training_data(
    n_points: int,
    *,
    seed: int,
    use_gee: bool,
    kalischek_asset: str,
    project: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N,64], labels [N])."""
    if use_gee:
        try:
            import ee

            from data.alphaearth_embeddings import AlphaEarthIngest
            from data.gee_auth import initialize_earth_engine

            initialize_earth_engine(project=project)
            ref = GeeKalischekReference(asset=kalischek_asset)
        except Exception as exc:
            logger.warning("GEE sampling failed (%s); using synthetic data", exc)
            return _synthetic_embeddings_and_labels(n_points, seed=seed)
    else:
        ref = HeuristicKalischekReference()

    rng = np.random.default_rng(seed)
    per_region = max(1, n_points // len(REGIONS))
    lats_list: list[np.ndarray] = []
    lons_list: list[np.ndarray] = []
    for i, region in enumerate(REGIONS):
        la, lo = _sample_grid(region, per_region, seed=seed + i)
        lats_list.append(la)
        lons_list.append(lo)
    lats = np.concatenate(lats_list)[:n_points]
    lons = np.concatenate(lons_list)[:n_points]

    if use_gee:
        import ee

        from data.alphaearth_embeddings import AlphaEarthIngest

        aoi = ee.Geometry.Rectangle([-9, 4, 2, 11])
        ingest = AlphaEarthIngest(aoi, year=2023, project=project)
        features = [
            ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"id": int(i)})
            for i, (la, lo) in enumerate(zip(lats, lons, strict=True))
        ]
        fc = ingest.sample_points(ee.FeatureCollection(features))
        rows = fc.getInfo()["features"]
        embeddings = np.array(
            [
                [float(f["properties"].get(b, 0.0)) for b in AEF_BAND_NAMES]
                for f in rows
            ],
            dtype=np.float32,
        )
        labels = ref.sample_reference(lats, lons).astype(np.float32)
        valid = np.isfinite(embeddings).all(axis=1)
        return embeddings[valid], (labels[valid] >= 0.5).astype(np.float32)

    # Heuristic path: synthetic embeddings from label + location seed
    labels = (ref.sample_reference(lats, lons) >= 0.5).astype(np.float32)
    embeddings = []
    for i, (la, lo, lab) in enumerate(zip(lats, lons, labels, strict=True)):
        rng_i = np.random.default_rng(int(hash((round(la, 3), round(lo, 3), i)) % (2**32)))
        vec = rng_i.normal(0, 1, AEF_EMBEDDING_DIM).astype(np.float32)
        if lab > 0.5:
            vec += 0.25
        vec /= np.linalg.norm(vec) + 1e-8
        embeddings.append(vec)
    return np.stack(embeddings, axis=0), labels


def train_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, float]:
    n = len(labels)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train = torch.from_numpy(embeddings[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    x_val = torch.from_numpy(embeddings[val_idx])
    y_val = torch.from_numpy(labels[val_idx])

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    model = AEFCocoaHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mlflow.set_experiment("aef_cocoa_head")
    with mlflow.start_run(run_name="aef_cocoa_head"):
        mlflow.log_params({"epochs": epochs, "n_train": len(train_idx), "n_val": len(val_idx)})
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.bce_loss(xb, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            if epoch % 5 == 0 or epoch == epochs - 1:
                mlflow.log_metric("train_bce", epoch_loss / max(n_batches, 1), step=epoch)

        model.eval()
        with torch.no_grad():
            val_prob = model.predict_proba(x_val.to(device)).cpu().numpy()
        val_acc = float(np.mean((val_prob >= 0.5) == (y_val.numpy() >= 0.5)))
        val_bce = float(
            model.bce_loss(x_val.to(device), y_val.to(device)).item()
        )
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_bce", val_bce)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "embedding_dim": AEF_EMBEDDING_DIM,
                "val_accuracy": val_acc,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(str(checkpoint_path))

    logger.info("Saved AEF head to %s (val_acc=%.3f)", checkpoint_path, val_acc)
    return {"val_accuracy": val_acc, "val_bce": val_bce}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train AEF cocoa classification head")
    parser.add_argument("--n-points", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gee", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--kalischek-asset", type=str, default=DEFAULT_KALISCHEK_ASSET)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_AEF_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        emb, lab = _synthetic_embeddings_and_labels(args.n_points, seed=args.seed)
    else:
        emb, lab = _sample_training_data(
            args.n_points,
            seed=args.seed,
            use_gee=args.use_gee,
            kalischek_asset=args.kalischek_asset,
            project=args.project,
        )

    train_head(
        emb,
        lab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
