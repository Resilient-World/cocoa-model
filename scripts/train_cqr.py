#!/usr/bin/env python3
"""
Train Conformalized Quantile Regression (CQR) yield model.

Uses the ICCO + CRIG panel (same source as ``training.train_yield``), with a
70/15/15 train/calibration/test split. Saves ``models/cqr_yield.pt`` and
``models/cqr_calibrator.joblib``.

Example::

    python scripts/train_cqr.py --max-epochs 30
    python scripts/train_cqr.py --synthetic --max-epochs 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.yield_panel import PanelRow, YieldPanelDataset, build_yield_panel
from models.cqr import (
    DEFAULT_CQR_CALIBRATOR,
    DEFAULT_CQR_CHECKPOINT,
    ConformalCalibrator,
    QuantileYieldSurrogate,
    pinball_loss,
)

logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
CAL_FRAC = 0.15


def _split_panel(
    rows: list[PanelRow],
    *,
    seed: int,
) -> tuple[list[PanelRow], list[PanelRow], list[PanelRow]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n = len(rows)
    n_train = int(n * TRAIN_FRAC)
    n_cal = int(n * CAL_FRAC)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train : n_train + n_cal]
    test_idx = idx[n_train + n_cal :]
    by_idx = [rows[i] for i in range(n)]
    return (
        [by_idx[i] for i in train_idx],
        [by_idx[i] for i in cal_idx],
        [by_idx[i] for i in test_idx],
    )


def _rows_to_tensors(rows: list[PanelRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    climates = np.stack([r.climate for r in rows], axis=0).astype(np.float32)
    statics = np.stack([r.static for r in rows], axis=0).astype(np.float32)
    targets = np.array([r.yield_target_pre_biotic_t_ha for r in rows], dtype=np.float32)
    return (
        torch.from_numpy(climates),
        torch.from_numpy(statics),
        torch.from_numpy(targets),
    )


@torch.no_grad()
def _evaluate_coverage(
    model: QuantileYieldSurrogate,
    calibrator: ConformalCalibrator,
    climate: torch.Tensor,
    static: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    lowers, _, uppers = calibrator.predict_interval_batch(
        model,
        (climate, static),
        device=device,
    )
    y_np = y.cpu().numpy()
    coverage = calibrator.empirical_coverage_on(y_np, lowers, uppers)
    width = float(np.mean(uppers - lowers))
    return {"coverage": coverage, "mean_interval_width": width}


def train_cqr(
    *,
    train_rows: list[PanelRow],
    cal_rows: list[PanelRow],
    test_rows: list[PanelRow],
    max_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    alpha: float = 0.2,
    device: torch.device,
    checkpoint_path: Path = DEFAULT_CQR_CHECKPOINT,
    calibrator_path: Path = DEFAULT_CQR_CALIBRATOR,
    mlflow_experiment: str = "cqr_yield",
    cv_strategy: str = "spatial_block",
    block_size_km: float = 50.0,
    variogram_from_checkpoint: bool = False,
) -> dict[str, float]:
    """Train quantile model, fit conformal calibrator, return test metrics."""
    train_loader = DataLoader(
        YieldPanelDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
    )
    model = QuantileYieldSurrogate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name="cqr_yield"):
        mlflow.log_params(
            {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "alpha": alpha,
                "n_train": len(train_rows),
                "n_cal": len(cal_rows),
                "n_test": len(test_rows),
            }
        )

        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(climate, static)
                loss = pinball_loss(pred, target, quantiles=model.quantiles)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            scheduler.step()
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                mlflow.log_metric("train_pinball", epoch_loss / max(n_batches, 1), step=epoch)

        all_rows = train_rows + cal_rows + test_rows
        climate_all, static_all, y_all = _rows_to_tensors(all_rows)
        lats = np.array(
            [
                {"GHA": 6.5, "CIV": 6.8, "CMR": 4.5, "NGA": 6.5}.get(r.country_iso3, 6.0)
                for r in all_rows
            ]
        )
        lons = np.array(
            [
                {"GHA": -1.5, "CIV": -5.5, "CMR": 9.5, "NGA": 5.5}.get(r.country_iso3, -2.0)
                for r in all_rows
            ]
        )
        block_km = block_size_km
        if variogram_from_checkpoint:
            from validation.spatial_cv import compute_residual_variogram, recommend_block_size_km

            with torch.no_grad():
                pred = model(climate_all.to(device), static_all.to(device)).cpu().numpy()[:, 1]
            res = y_all.cpu().numpy() - pred
            vario = compute_residual_variogram(
                pred,
                res,
                np.column_stack([lons, lats]),
            )
            block_km = recommend_block_size_km(vario["range_km"])
            logger.info("Variogram range_km=%.1f → block_size_km=%.1f", vario["range_km"], block_km)

        calibrator = ConformalCalibrator()
        if cv_strategy == "spatial_block":
            from validation.spatial_cv import SpatialBlockSplit

            splitter = SpatialBlockSplit(block_size_km=block_km, n_folds=5, seed=42)
            calibrator.fit_blocked(
                model,
                (climate_all, static_all),
                y_all,
                splitter,
                coords=(lats, lons),
                alpha=alpha,
                device=device,
            )
        else:
            cal_climate, cal_static, cal_y = _rows_to_tensors(cal_rows)
            calibrator.fit(
                model,
                (cal_climate, cal_static),
                cal_y,
                alpha=alpha,
                device=device,
            )

        test_climate, test_static, test_y = _rows_to_tensors(test_rows)
        test_metrics = _evaluate_coverage(
            model,
            calibrator,
            test_climate,
            test_static,
            test_y,
            device=device,
        )
        mlflow.log_metric("test_coverage", test_metrics["coverage"])
        mlflow.log_metric("test_mean_interval_width", test_metrics["mean_interval_width"])
        mlflow.log_metric("calibration_coverage", float(calibrator.empirical_coverage or 0.0))
        mlflow.log_metric("Q_hat", float(calibrator.Q_hat or 0.0))

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "quantiles": model.quantiles,
                "alpha": alpha,
            },
            checkpoint_path,
        )
        calibrator.save(calibrator_path)
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(calibrator_path))

    logger.info(
        "CQR test coverage=%.3f width=%.3f (nominal %.0f%%)",
        test_metrics["coverage"],
        test_metrics["mean_interval_width"],
        (1.0 - alpha) * 100,
    )
    return test_metrics


def _synthetic_panel(n: int = 600, *, seed: int = 0) -> list[PanelRow]:
    """Synthetic panel for CI when ICCO CSVs are absent."""
    from data.yield_panel import PanelRow

    rng = np.random.default_rng(seed)
    rows: list[PanelRow] = []
    for i in range(n):
        climate = rng.normal(0, 0.1, (365, 11)).astype(np.float32)
        climate[:, 0] = 28 + rng.normal(0, 1, 365)
        climate[:, 1] = 22 + rng.normal(0, 1, 365)
        climate[:, 2] = 25 + rng.normal(0, 1, 365)
        climate[:, 3] = np.clip(rng.gamma(2, 2, 365), 0, 40)
        climate[:, 4] = 15
        climate[:, 5] = 1.0
        climate[:, 6] = 3.5
        climate[:, 7] = 0.3
        climate[:, 8] = 2.0
        climate[:, 9] = 80
        climate[:, 10] = 415
        static = rng.normal(0, 0.1, 13).astype(np.float32)
        static[0] = 150.0
        signal = float(static[0] * 0.005 + climate[:, 2].mean() * 0.04 + rng.normal(0, 0.15))
        y = max(0.2, signal)
        rows.append(
            PanelRow(
                sample_id=f"syn_{i}",
                country_iso3="GHA",
                year=2020 + (i % 4),
                cohort="icco",
                yield_observed_t_ha=y,
                yield_target_pre_biotic_t_ha=y,
                surviving_biotic_fraction=1.0,
                climate=climate,
                static=static,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CQR yield surrogate")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="1-alpha = nominal coverage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic panel (CI)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CQR_CHECKPOINT)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CQR_CALIBRATOR)
    parser.add_argument(
        "--cv-strategy",
        choices=("random", "spatial_block"),
        default="spatial_block",
        help="Conformal calibration split (production: spatial_block)",
    )
    parser.add_argument("--block-size-km", type=float, default=50.0)
    parser.add_argument(
        "--variogram-from-checkpoint",
        action="store_true",
        help="Set block size from residual variogram × 1.5 (Roberts 2017)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        rows = _synthetic_panel(900, seed=args.seed)
    else:
        try:
            rows = build_yield_panel(seed=args.seed)
        except FileNotFoundError:
            logger.warning("ICCO panel missing; falling back to synthetic data")
            rows = _synthetic_panel(900, seed=args.seed)

    train_rows, cal_rows, test_rows = _split_panel(rows, seed=args.seed)
    metrics = train_cqr(
        train_rows=train_rows,
        cal_rows=cal_rows,
        test_rows=test_rows,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        device=device,
        checkpoint_path=args.checkpoint,
        calibrator_path=args.calibrator,
        cv_strategy=args.cv_strategy,
        block_size_km=args.block_size_km,
        variogram_from_checkpoint=args.variogram_from_checkpoint,
    )
    nominal = 1.0 - args.alpha
    if metrics["coverage"] < nominal - 0.02:
        logger.warning(
            "Test coverage %.3f below nominal %.0f%% target",
            metrics["coverage"],
            nominal * 100,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
