#!/usr/bin/env python3
"""
Fine-tune TerraMind 1.0 for binary cocoa segmentation.

BCE+Dice loss, hard-example mining, MLflow logging (experiment ``terramind_cocoa_finetune``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning.pytorch as pl
import structlog
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torchmetrics.classification import BinaryF1Score

from models.terramind_seg import (
    DEFAULT_TERRAMIND_TIM_CHECKPOINT,
    TerraMindCocoaSegmentation,
    TerraMindTiMCocoaSegmentation,
)
from training.agrifm_losses import agrifm_bce_dice_loss
from training.cocoa_terramind_datamodule import (
    CocoaTerraMindDataModule,
    SyntheticTerraMindDataModule,
)
from training.hard_example_mining import HardExampleMiningCallback

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "models" / "terramind_cocoa_seg.pt"


class UnfreezeTerraMindBackboneCallback(Callback):
    def __init__(self, unfreeze_epoch: int = 20) -> None:
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        model = getattr(pl_module, "model", None)
        if model is not None and hasattr(model, "set_backbone_freeze"):
            model.set_backbone_freeze(False)


class TerraMindCocoaTask(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 6e-5,
        weight_decay: float = 0.01,
        pos_weight: float = 4.0,
        max_epochs: int = 50,
        use_tim: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.use_tim = use_tim
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_tim:
            return self.model.forward_dict(batch)
        x = batch["terramind"]
        feat = self.model.backbone(x)
        return self.model.head(feat)

    def _step(self, batch: dict[str, torch.Tensor], step: str) -> torch.Tensor:
        if "terramind" not in batch:
            from data.utils import cocoa_batch_to_terramind_input

            batch = {**batch, "terramind": cocoa_batch_to_terramind_input(batch)}
        if self.use_tim:
            logits = self.model.forward_dict(batch)
        else:
            logits = self.forward(batch)
        target = batch["target"]
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = agrifm_bce_dice_loss(logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long().squeeze(1)
        tgt = (target.squeeze(1) >= 0.5).long()
        if step == "train":
            self.train_f1.update(pred, tgt)
        else:
            self.val_f1.update(pred, tgt)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self) -> dict[str, object]:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def export_checkpoint(pl_module: TerraMindCocoaTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": pl_module.model.state_dict()}, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TerraMind for cocoa segmentation")
    p.add_argument("--image-paths", type=Path, default=Path("data/processed/images"))
    p.add_argument("--mask-paths", type=Path, default=Path("data/processed/masks"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tim", action="store_true", help="Train TiM re-encode path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--freeze-epochs", type=int, default=20)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-tiles", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mlflow-experiment", type=str, default="terramind_cocoa_finetune")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints/terramind"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.freeze_epochs = 0
        args.patch_size = 64
        args.batch_size = 2

    pl.seed_everything(42, workers=True)

    if args.synthetic:
        n_val = max(10, args.max_tiles // 5)
        dm: SyntheticTerraMindDataModule | CocoaTerraMindDataModule = SyntheticTerraMindDataModule(
            n_train=args.max_tiles - n_val,
            n_val=n_val,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
        )
    else:
        dm = CocoaTerraMindDataModule(
            image_paths=args.image_paths,
            mask_paths=args.mask_paths,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            length=500,
        )

    if args.tim:
        seg = TerraMindTiMCocoaSegmentation()
        out_path = DEFAULT_TERRAMIND_TIM_CHECKPOINT if args.out == DEFAULT_OUT else args.out
    else:
        seg = TerraMindCocoaSegmentation(freeze_backbone=True)
        out_path = args.out

    task = TerraMindCocoaTask(seg, max_epochs=args.epochs, use_tim=args.tim)
    logger = MLFlowLogger(experiment_name=args.mlflow_experiment, run_name="terramind-cocoa-seg")
    logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=str(args.checkpoint_dir), monitor="val/f1", mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
            UnfreezeTerraMindBackboneCallback(args.freeze_epochs),
            HardExampleMiningCallback(),
        ],
        log_every_n_steps=5,
    )
    try:
        trainer.fit(task, datamodule=dm)
    except FileNotFoundError as exc:
        log.info(
            f"Missing tiles ({exc}); use --synthetic",
        )
        return 1
    export_checkpoint(task, out_path)
    log.info(f"Exported TerraMind checkpoint → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
