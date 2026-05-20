#!/usr/bin/env python3
"""
Train :class:`models.joint_exposure_yield.JointHead` with PyTorch Lightning + MLflow.

Shared backbone feature maps (synthetic or precomputed embeddings) drive joint
segmentation + yield (+ CQR pinball) losses.

Example::

    python scripts/train_joint.py --synthetic --epochs 10
    python scripts/train_joint.py --epochs 30 --max-epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.joint_exposure_yield import (
    DEFAULT_LAMBDA_CQR,
    JointHead,
    JointMultiTaskLoss,
    JointOutputs,
)
from models.yield_surrogate import N_STATIC_SITE

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "joint.pt"
DEFAULT_BACKBONE_DIM = 128
DEFAULT_MAP_SIZE = 16


class JointTileDataset(Dataset[dict[str, Tensor]]):
    """Synthetic or parquet-backed tiles for joint training."""

    def __init__(
        self,
        n_samples: int,
        *,
        seed: int = 42,
        backbone_dim: int = DEFAULT_BACKBONE_DIM,
        map_size: int = DEFAULT_MAP_SIZE,
        static_dim: int = N_STATIC_SITE,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.backbone_dim = backbone_dim
        self.map_size = map_size
        self.static_dim = static_dim

        self.features = rng.normal(0, 1, (n_samples, backbone_dim, map_size, map_size)).astype(
            np.float32
        )
        self.static = rng.uniform(0, 1, (n_samples, static_dim)).astype(np.float32)
        self.static[:, 0] = rng.uniform(80, 200, n_samples)
        # Seg mask correlated with mean feature
        score = self.features.mean(axis=(1, 2, 3))
        self.seg = (score > np.median(score)).astype(np.float32)[:, None, None, None]
        self.seg = np.broadcast_to(self.seg, (n_samples, 1, map_size, map_size)).copy()
        self.yield_t = (
            0.8
            + 0.6 * self.seg[:, 0, 0, 0]
            + 0.15 * self.static[:, 1]
            + rng.normal(0, 0.1, n_samples)
        ).astype(np.float32)
        self.yield_t = np.clip(self.yield_t, 0.2, 4.0)

    def __len__(self) -> int:
        return len(self.yield_t)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "backbone": torch.from_numpy(self.features[idx]),
            "static": torch.from_numpy(self.static[idx]),
            "seg": torch.from_numpy(self.seg[idx]),
            "yield": torch.tensor(self.yield_t[idx], dtype=torch.float32),
        }


class JointLightningModule(pl.LightningModule):
    """Lightning wrapper for :class:`JointHead` + :class:`JointMultiTaskLoss`."""

    def __init__(
        self,
        *,
        backbone_dim: int = DEFAULT_BACKBONE_DIM,
        static_dim: int = N_STATIC_SITE,
        hidden_dim: int = 256,
        lambda_cqr: float = DEFAULT_LAMBDA_CQR,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.head = JointHead(
            backbone_dim=backbone_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
        )
        self.loss_fn = JointMultiTaskLoss(lambda_cqr=lambda_cqr)

    def forward(self, backbone: Tensor, static: Tensor) -> JointOutputs:
        return self.head(backbone, static)

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self(batch["backbone"], batch["static"])
        loss, breakdown = self.loss_fn(
            out,
            seg_target=batch["seg"],
            yield_target=batch["yield"],
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_seg_bce", breakdown.seg_bce, on_step=False, on_epoch=True)
        self.log(f"{stage}_yield_mse", breakdown.yield_mse, on_step=False, on_epoch=True)
        self.log(f"{stage}_yield_pinball", breakdown.yield_pinball, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def train_joint(
    *,
    train_ds: Dataset,
    val_ds: Dataset,
    max_epochs: int,
    batch_size: int,
    checkpoint_path: Path,
    accelerator: str = "auto",
) -> Path:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    module = JointLightningModule()
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_path.parent,
        filename="joint-epoch{epoch:02d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    csv_logger = CSVLogger(save_dir=str(checkpoint_path.parent / "joint_logs"), name="joint")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=csv_logger,
        callbacks=[ckpt_cb, early],
        enable_checkpointing=True,
        log_every_n_steps=5,
    )
    trainer.fit(module, train_loader, val_loader)

    best = ckpt_cb.best_model_path
    state = module.state_dict()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "config": {
                "backbone_dim": module.head.backbone_dim,
                "static_dim": module.head.static_dim,
                "lambda_cqr": module.loss_fn.lambda_cqr,
            },
            "best_ckpt": best,
        },
        checkpoint_path,
    )

    mlruns = (checkpoint_path.parent / "mlruns").resolve()
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns.as_uri())
    exp = mlflow.get_experiment_by_name("joint_exposure_yield")
    if exp is None:
        mlflow.create_experiment("joint_exposure_yield")
    mlflow.set_experiment("joint_exposure_yield")
    with mlflow.start_run(run_name="train_joint"):
        mlflow.log_params(
            {
                "max_epochs": max_epochs,
                "backbone_dim": module.head.backbone_dim,
                "static_dim": module.head.static_dim,
                "lambda_cqr": module.loss_fn.lambda_cqr,
            }
        )
        if trainer.callback_metrics:
            for key, val in trainer.callback_metrics.items():
                if hasattr(val, "item"):
                    mlflow.log_metric(key.replace("/", "_"), float(val.item()))
        mlflow.log_artifact(str(checkpoint_path))

    logger.info("Saved joint head to %s (best=%s)", checkpoint_path, best)
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train joint exposure + yield head")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic tiles (default)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_ds = JointTileDataset(args.n_train, seed=args.seed)
    val_ds = JointTileDataset(args.n_val, seed=args.seed + 1)
    train_joint(
        train_ds=train_ds,
        val_ds=val_ds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
