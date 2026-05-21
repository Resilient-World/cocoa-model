#!/usr/bin/env python3
"""
Fine-tune AgriFM Video Swin for binary cocoa segmentation.

Mirrors :mod:`training.train_galileo_cocoa` with BCE+Dice loss, hard-example mining,
and MLflow experiment ``agrifm_cocoa_finetune``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torchmetrics.classification import BinaryF1Score

from models.agrifm_seg import AgriFMCocoaSegmentation, DEFAULT_AGRIFM_CHECKPOINT
from training.agrifm_losses import agrifm_bce_dice_loss
from training.cocoa_agrifm_datamodule import (
    CocoaAgriFMDataModule,
    SyntheticAgriFMDataModule,
)
from training.hard_example_mining import HardExampleMiningCallback

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"
DEFAULT_PRETRAINED = _REPO_ROOT / "models" / "agrifm" / "agrifm_s2_pretrained.pt"


class UnfreezeAgriFMBackboneCallback(Callback):
    """Unfreeze encoder after ``unfreeze_epoch`` (two-stage fine-tuning)."""

    def __init__(self, unfreeze_epoch: int = 20) -> None:
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        if hasattr(pl_module, "model") and hasattr(pl_module.model, "set_backbone_freeze"):
            pl_module.model.set_backbone_freeze(False)
            trainer.logger.log_metrics(
                {"agrifm/backbone_frozen": 0.0},
                step=trainer.global_step,
            )


class AgriFMCocoaTask(pl.LightningModule):
    """PyTorch Lightning module for AgriFM cocoa segmentation."""

    def __init__(
        self,
        model: AgriFMCocoaSegmentation,
        *,
        lr: float = 6e-5,
        weight_decay: float = 0.01,
        pos_weight: float = 4.0,
        max_epochs: int = 50,
        hard_loss_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.hard_loss_threshold = hard_loss_threshold
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self._per_sample_losses: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], step: str) -> torch.Tensor:
        logits = self.forward(batch["agrifm"])
        target = batch["target"]
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = agrifm_bce_dice_loss(logits, target, pos_weight=self.pos_weight)
        if (
            step == "train"
            and self.hard_loss_threshold is not None
            and float(loss.detach()) > self.hard_loss_threshold
        ):
            loss = loss * 1.5
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long().squeeze(1)
        tgt = (target.squeeze(1) >= 0.5).long()
        if step == "train":
            self.train_f1.update(pred, tgt)
        else:
            self.val_f1.update(pred, tgt)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def export_checkpoint(
    pl_module: AgriFMCocoaTask,
    path: Path,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    """Save full segmentation weights for inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "state_dict": pl_module.model.state_dict(),
        "out_size": list(pl_module.model.out_size),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AgriFM for cocoa segmentation")
    parser.add_argument("--image-paths", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--mask-paths", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-length", type=int, default=500)
    parser.add_argument("--synthetic", action="store_true", help="100-tile in-memory synthetic run")
    parser.add_argument("--max-tiles", type=int, default=100, help="Synthetic train+val tile count")
    parser.add_argument("--quick", action="store_true", help="2 epochs, small patches")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="agrifm_cocoa_finetune",
    )
    parser.add_argument("--mlflow-run-name", type=str, default="agrifm-cocoa-seg")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints/agrifm"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.freeze_epochs = 0
        args.patch_size = 64
        args.train_length = 20
        args.batch_size = 2

    pl.seed_everything(args.seed, workers=True)

    if args.synthetic:
        n_val = max(10, args.max_tiles // 5)
        n_train = args.max_tiles - n_val
        datamodule: SyntheticAgriFMDataModule | CocoaAgriFMDataModule = SyntheticAgriFMDataModule(
            n_train=n_train,
            n_val=n_val,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            seed=args.seed,
        )
    else:
        datamodule = CocoaAgriFMDataModule(
            image_paths=args.image_paths,
            mask_paths=args.mask_paths,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            length=args.train_length,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            seed=args.seed,
        )

    pretrained = args.pretrained if args.pretrained.is_file() else DEFAULT_AGRIFM_CHECKPOINT
    model = AgriFMCocoaSegmentation(
        checkpoint_path=pretrained,
        freeze_backbone=True,
        out_size=(args.patch_size, args.patch_size),
        num_frames=args.num_frames,
    )

    task = AgriFMCocoaTask(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        max_epochs=args.epochs,
    )

    if args.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", args.mlflow_tracking_uri)

    logger = MLFlowLogger(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        log_model="all",
    )
    logger.log_hyperparams(vars(args))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="agrifm-cocoa-{epoch:02d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
    )

    callbacks: list[Callback] = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        UnfreezeAgriFMBackboneCallback(unfreeze_epoch=args.freeze_epochs),
        HardExampleMiningCallback(),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(task, datamodule=datamodule)
    except FileNotFoundError as exc:
        print(f"Tile data missing ({exc}); re-run with --synthetic", file=sys.stderr)
        return 1

    export_checkpoint(task, args.out, extra={"epochs": args.epochs, "patch_size": args.patch_size})
    print(f"Exported AgriFM cocoa segmentation → {args.out}")
    if checkpoint_callback.best_model_path:
        print(f"Best Lightning checkpoint: {checkpoint_callback.best_model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
