#!/usr/bin/env python3
"""Benchmark full fine-tune vs LoRA adapter size on synthetic Ghana cocoa labels."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.lora_adapter import (  # type: ignore[import-untyped]
    apply_lora_to_backbone,
    save_lora_for_region,
    trainable_parameter_fraction,
)


@dataclass
class BenchmarkResult:
    f1: float
    miou: float
    checkpoint_mb: float
    train_seconds: float


class TinyGalileoCocoa(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def _synthetic_ghana(n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 32, generator=g)
    weights = torch.linspace(-1.0, 1.0, 32)
    logit = x @ weights + 0.35 * torch.sin(x[:, 0] * 2.0)
    y = (torch.sigmoid(logit) > 0.5).float()
    return x, y


def _score(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(x)) >= 0.5).float()
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
    miou = tp / max(tp + fp + fn, 1.0)
    return f1, miou


def _train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return time.perf_counter() - start


def _full_size(model: nn.Module, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def _run(args: argparse.Namespace) -> tuple[BenchmarkResult, BenchmarkResult, float]:
    x, y = _synthetic_ghana(args.n, seed=args.seed)
    full = TinyGalileoCocoa()
    full_seconds = _train(full, x, y, args.epochs)
    full_f1, full_miou = _score(full, x, y)
    full_mb = _full_size(full, args.out_dir / "galileo_full_ghana.pt")

    lora = TinyGalileoCocoa()
    lora.load_state_dict(full.state_dict())
    lora.backbone = apply_lora_to_backbone(
        lora.backbone,
        "aef",
        r=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=("0", "2"),
    )
    lora_seconds = _train(lora, x, y, args.epochs)
    lora_f1, lora_miou = _score(lora, x, y)
    adapter_path = save_lora_for_region(
        lora.backbone,
        "ghana",
        args.out_dir,
        backbone_name="galileo",
    )
    lora_mb = adapter_path.stat().st_size / (1024 * 1024)
    reduction = full_mb / max(lora_mb, 1e-9)
    return (
        BenchmarkResult(full_f1, full_miou, full_mb, full_seconds),
        BenchmarkResult(lora_f1, lora_miou, lora_mb, lora_seconds),
        reduction,
    )


def _write_report(
    path: Path,
    full: BenchmarkResult,
    lora: BenchmarkResult,
    reduction: float,
    trainable_fraction: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f1_loss_pp = (full.f1 - lora.f1) * 100.0
    status = "PASS" if reduction >= 10.0 and f1_loss_pp <= 2.0 else "REVIEW"
    path.write_text(
        "\n".join(
            [
                "# LoRA vs full fine-tune benchmark",
                "",
                f"Status: {status}",
                "",
                "| Mode | F1 | mIoU | Checkpoint MB | Train seconds |",
                "|---|---:|---:|---:|---:|",
                f"| Full fine-tune | {full.f1:.4f} | {full.miou:.4f} | {full.checkpoint_mb:.4f} | {full.train_seconds:.3f} |",
                f"| LoRA adapter | {lora.f1:.4f} | {lora.miou:.4f} | {lora.checkpoint_mb:.4f} | {lora.train_seconds:.3f} |",
                "",
                f"- Checkpoint reduction: {reduction:.1f}×",
                f"- F1 loss: {f1_loss_pp:.2f} pp",
                f"- LoRA trainable parameter fraction: {trainable_fraction:.4%}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="ghana")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("models/benchmarks"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/training"))
    args = parser.parse_args(argv)
    del args.region
    full, lora, reduction = _run(args)
    probe = TinyGalileoCocoa()
    probe.backbone = apply_lora_to_backbone(
        probe.backbone, "aef", r=4, alpha=8, target_modules=("0", "2")
    )
    report = args.report_dir / f"lora_vs_full_{datetime.now(UTC).strftime('%Y%m%d')}.md"
    _write_report(report, full, lora, reduction, trainable_parameter_fraction(probe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
