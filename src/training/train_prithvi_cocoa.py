#!/usr/bin/env python3
"""
Fine-tune NASA-IBM Prithvi (TerraTorch) for cocoa plantation semantic segmentation.

Uses CocoaPrithviDataModule (TorchGeo) + SemanticSegmentationTask (TerraTorch)
with MLflow logging, AdamW, and cosine LR scheduling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import structlog
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from terratorch.datasets import HLSBands
from terratorch.tasks import SemanticSegmentationTask

from data.cocoa_dataset import CLASS_NAMES
from training.cocoa_prithvi_datamodule import (
    PRITHVI_RGB_BANDS,
    PRITHVI_SENTINEL2_BANDS,
    CocoaPrithviDataModule,
    hls_bands_for_input,
)

log = structlog.get_logger(__name__)

NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_BACKBONE = "prithvi_eo_v2_100_tl"
DEFAULT_DECODER = "UperNetDecoder"  # FPN-style decoder; use "UNetDecoder" for U-Net


def build_prithvi_model_args(
    *,
    num_classes: int = NUM_CLASSES,
    backbone: str = DEFAULT_BACKBONE,
    decoder: str = DEFAULT_DECODER,
    backbone_bands: list[HLSBands],
    backbone_in_chans: int,
    pretrained: bool = True,
) -> dict:
    """
    Model configuration for TerraTorch EncoderDecoderFactory + Prithvi backbone.

    Input channels
    --------------
    Prithvi EO v2 was pretrained on **6** HLS-like bands (BLUE, GREEN, RED,
    NIR_NARROW, SWIR_1, SWIR_2). The default cocoa pipeline uses six Sentinel-2
    bands (B2, B3, B4, B8, B11, B12).

    If your GeoTIFFs only have **3 bands (RGB)**, change the DataModule to::

        input_bands=PRITHVI_RGB_BANDS  # ("B4", "B3", "B2")

    and pass matching arguments here::

        backbone_bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
        backbone_in_chans=3

    Extra bands (e.g. Sentinel-1 VV/VH) are not used by the Prithvi patch embed
    unless you re-init those channels (pretrained=False or custom band list).
    """
    # SelectIndices: intermediate ViT layers for 100M / tiny Prithvi (see TerraTorch docs)
    select_indices = [2, 5, 8, 11]
    if "300" in backbone:
        select_indices = [5, 11, 17, 23]
    elif "600" in backbone:
        select_indices = [7, 15, 23, 31]

    return {
        "task": "segmentation",
        "backbone": backbone,
        "backbone_pretrained": pretrained,
        "backbone_bands": backbone_bands,
        "backbone_in_chans": backbone_in_chans,
        "backbone_num_frames": 1,
        "decoder": decoder,
        "num_classes": num_classes,
        "necks": [
            {"name": "ReshapeTokensToImage"},
            {"name": "SelectIndices", "indices": select_indices},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi for cocoa segmentation")
    parser.add_argument(
        "--image-paths",
        "--image-dir",
        type=Path,
        default=Path("data/processed/images"),
        dest="image_paths",
    )
    parser.add_argument(
        "--mask-paths",
        "--mask-dir",
        type=Path,
        default=Path("data/processed/masks"),
        dest="mask_paths",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-length", type=int, default=500, help="Samples per training epoch")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument(
        "--decoder",
        type=str,
        default=DEFAULT_DECODER,
        choices=("UperNetDecoder", "UNetDecoder", "FCNDecoder"),
    )
    parser.add_argument(
        "--input-bands",
        nargs="+",
        default=list(PRITHVI_SENTINEL2_BANDS),
        help=(
            "Imagery bands to feed Prithvi. Default: 6 S2 bands matching pretraining. "
            "For RGB only use: B4 B3 B2"
        ),
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="resilient-cocoa-model")
    parser.add_argument("--mlflow-run-name", type=str, default="prithvi-cocoa-seg")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory for Lightning checkpoints",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Copy best checkpoint to this path (e.g. models/segmentation.ckpt)",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    input_bands = tuple(args.input_bands)
    if len(input_bands) == 3:
        input_bands = PRITHVI_RGB_BANDS
    elif len(input_bands) != 6:
        log.error(
            "invalid_prithvi_bands",
            expected_6=list(PRITHVI_SENTINEL2_BANDS),
            expected_3=list(PRITHVI_RGB_BANDS),
        )
        return 1

    datamodule = CocoaPrithviDataModule(
        image_paths=args.image_paths,
        mask_paths=args.mask_paths,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        length=args.train_length,
        num_workers=args.num_workers,
        input_bands=input_bands,
        seed=args.seed,
    )

    try:
        backbone_bands = hls_bands_for_input(input_bands)
    except ValueError as exc:
        log.info(
            "Unsupported --input-bands. Use 6-band Prithvi S2 stack "
            f"{PRITHVI_SENTINEL2_BANDS} or RGB {PRITHVI_RGB_BANDS}.",
        )
        log.error("train_prithvi_failed", error=str(exc))
        return 1

    model_args = build_prithvi_model_args(
        backbone=args.backbone,
        decoder=args.decoder,
        backbone_bands=backbone_bands,
        backbone_in_chans=len(backbone_bands),
        pretrained=not args.no_pretrained,
    )

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        loss="ce",
        ignore_index=None,
        class_names=datamodule.class_names,
        lr=args.lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": args.weight_decay},
        scheduler="CosineAnnealingLR",
        scheduler_hparams={"T_max": args.epochs},
        plot_on_val=5,
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
            "model_args": model_args,
            "input_bands": list(input_bands),
        }
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="prithvi-cocoa-{epoch:02d}-{val/mIoU:.4f}",
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
        ],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    log.info(
        f"Training Prithvi ({args.backbone} + {args.decoder}) "
        f"with {len(backbone_bands)} input channels, {NUM_CLASSES} classes, "
        f"{args.epochs} epochs."
    )
    trainer.fit(task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path
    log.info(f"Best checkpoint: {best_path}")
    if args.out is not None and best_path:
        import shutil

        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.out)
        log.info(f"Copied best checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
