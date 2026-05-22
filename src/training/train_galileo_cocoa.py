#!/usr/bin/env python3
"""
Fine-tune Galileo (NASA Harvest) for cocoa plantation semantic segmentation.

Mirrors :mod:`training.train_prithvi_cocoa` but uses :class:`~models.galileo_backbone.GalileoSegmentation`
with inverse-frequency class weights, two-stage backbone freezing, and MLflow logging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import structlog
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torchmetrics.classification import MulticlassJaccardIndex

from data.cocoa_dataset import CLASS_NAMES, CLASS_OTHER
from models.galileo_backbone import DEFAULT_NUM_CLASSES, GalileoSegmentation
from training.cocoa_galileo_datamodule import CocoaGalileoDataModule

log = structlog.get_logger(__name__)

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = CLASS_OTHER  # presence-only background (FTW-style)


def inverse_frequency_weights(
    counts: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = IGNORE_INDEX,
) -> torch.Tensor:
    """Class weights inversely proportional to pixel frequency."""
    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue
        freq = counts[class_id].float().clamp(min=1.0)
        weights[class_id] = 1.0 / freq
    active = [i for i in range(num_classes) if ignore_index is None or i != ignore_index]
    if active:
        subset = weights[active]
        weights[active] = subset / subset.sum() * len(active)
    return weights


def estimate_class_weights(
    datamodule: CocoaGalileoDataModule,
    *,
    max_batches: int = 50,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Estimate inverse-frequency weights from training batches."""
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        mask = batch["mask"]
        for class_id in range(NUM_CLASSES):
            if class_id == ignore_index:
                continue
            counts[class_id] += (mask == class_id).sum().item()
    return inverse_frequency_weights(counts, ignore_index=ignore_index)


class UnfreezeGalileoBackboneCallback(Callback):
    """Unfreeze the Galileo encoder after ``unfreeze_epoch`` (two-stage fine-tuning)."""

    def __init__(self, unfreeze_epoch: int = 20) -> None:
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        if isinstance(pl_module, GalileoCocoaTask):
            pl_module.unfreeze_backbone()
            trainer.logger.log_metrics(
                {"galileo/backbone_frozen": 0.0},
                step=trainer.global_step,
            )


class GalileoCocoaTask(pl.LightningModule):
    """PyTorch Lightning module for Galileo cocoa segmentation."""

    def __init__(
        self,
        model: GalileoSegmentation,
        *,
        class_weights: torch.Tensor | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        ignore_index: int = IGNORE_INDEX,
        unfreeze_epoch: int = 20,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.unfreeze_epoch = unfreeze_epoch
        self.max_epochs = max_epochs
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(NUM_CLASSES),
            persistent=False,
        )
        self.train_iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES,
            ignore_index=ignore_index,
            average="macro",
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES,
            ignore_index=ignore_index,
            average="macro",
        )

    def unfreeze_backbone(self) -> None:
        self.model.set_backbone_freeze(False)

    def forward(self, galileo_batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        return self.model(galileo_batch)

    def _shared_step(self, batch: dict[str, torch.Tensor], step: str) -> torch.Tensor:
        logits = self.forward(batch["galileo"])
        target = batch["mask"]
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )
        preds = logits.argmax(dim=1)
        if step == "train":
            self.train_iou.update(preds, target)
        else:
            self.val_iou.update(preds, target)
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
        self.log("train/mIoU", self.train_iou.compute(), prog_bar=True)
        self.train_iou.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/mIoU", self.val_iou.compute(), prog_bar=True)
        self.val_iou.reset()

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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Galileo for cocoa segmentation")
    parser.add_argument("--image-paths", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--mask-paths", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--srtm-paths", type=Path, default=None)
    parser.add_argument("--era5-paths", type=Path, default=None)
    parser.add_argument("--terraclim-paths", type=Path, default=None)
    parser.add_argument("--dynamic-world-paths", type=Path, default=None)
    parser.add_argument("--world-cereal-paths", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-length", type=int, default=500)
    parser.add_argument("--model-size", type=str, default="base", choices=("nano", "tiny", "base"))
    parser.add_argument("--galileo-patch-size", type=int, default=4)
    parser.add_argument("--decoder", type=str, default="upernet", choices=("upernet", "fpn"))
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="resilient-cocoa-model")
    parser.add_argument("--mlflow-run-name", type=str, default="galileo-cocoa-seg")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--class-weight-batches", type=int, default=50)
    parser.add_argument(
        "--synthetic", action="store_true", help="Smoke training without tile files"
    )
    parser.add_argument("--quick", action="store_true", help="Few epochs / nano model for CI")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Copy best weights to this path (e.g. models/galileo_cocoa_seg.pt)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Write training metrics JSON for DVC",
    )
    return parser.parse_args(argv)


def _write_smoke_galileo_checkpoint(out: Path, *, model_size: str = "nano") -> None:
    """Save minimal Galileo seg weights for offline DVC repro."""
    from models.galileo_seg import GalileoCocoaSegmentation

    out.parent.mkdir(parents=True, exist_ok=True)
    model = GalileoCocoaSegmentation(model_size=model_size, freeze_backbone=True)
    torch.save({"state_dict": model.state_dict(), "smoke": True, "model_size": model_size}, out)
    log.info("smoke_galileo_checkpoint", path=str(out))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.freeze_epochs = 0
        args.model_size = "nano"
        args.train_length = min(args.train_length, 32)
        args.batch_size = min(args.batch_size, 2)
        args.class_weight_batches = 2

    if args.synthetic:
        out = args.out or Path("models/galileo_cocoa_seg.pt")
        _write_smoke_galileo_checkpoint(out, model_size=args.model_size)
        if args.metrics_out:
            import json

            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(
                json.dumps({"smoke": True, "val_mIoU": 0.0}),
                encoding="utf-8",
            )
        return 0

    pl.seed_everything(args.seed, workers=True)

    datamodule = CocoaGalileoDataModule(
        image_paths=args.image_paths,
        mask_paths=args.mask_paths,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        length=args.train_length,
        num_workers=args.num_workers,
        srtm_paths=args.srtm_paths,
        era5_paths=args.era5_paths,
        terraclim_paths=args.terraclim_paths,
        dynamic_world_paths=args.dynamic_world_paths,
        world_cereal_paths=args.world_cereal_paths,
        seed=args.seed,
    )

    try:
        class_weights = estimate_class_weights(
            datamodule,
            max_batches=args.class_weight_batches,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning(
            "class_weight_estimation_failed",
            error=str(exc),
            fallback="uniform",
        )
        class_weights = torch.ones(NUM_CLASSES)

    segmentation = GalileoSegmentation(
        model_size=args.model_size,
        num_classes=DEFAULT_NUM_CLASSES,
        patch_size=args.galileo_patch_size,
        freeze_backbone=True,
        decoder=args.decoder,
    )

    task = GalileoCocoaTask(
        segmentation,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_epoch=args.freeze_epochs,
        max_epochs=args.epochs,
    )

    if args.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", args.mlflow_tracking_uri)

    logger = MLFlowLogger(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        log_model="all",
    )
    logger.log_hyperparams(
        {
            **vars(args),
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "class_weights": class_weights.tolist(),
        }
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="galileo-cocoa-{epoch:02d}-{val/mIoU:.4f}",
        monitor="val/mIoU",
        mode="max",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            UnfreezeGalileoBackboneCallback(unfreeze_epoch=args.freeze_epochs),
        ],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    log.info(
        f"Training Galileo ({args.model_size}, decoder={args.decoder}) "
        f"with {NUM_CLASSES} classes (ignore_index={IGNORE_INDEX}), "
        f"freeze backbone for {args.freeze_epochs} epochs, {args.epochs} total epochs."
    )
    trainer.fit(task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule)

    log.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    if args.out is not None:
        import shutil

        best = checkpoint_callback.best_model_path
        if best:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, args.out)
            log.info("copied_galileo_checkpoint", dest=str(args.out))
    if args.metrics_out is not None:
        import json

        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps({"best_checkpoint": checkpoint_callback.best_model_path}),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
