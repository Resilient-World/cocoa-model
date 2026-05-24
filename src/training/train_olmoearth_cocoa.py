"""Fine-tune OlmoEarth cocoa segmentation head on Kalischek tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.olmoearth_seg import OlmoEarthCocoaSegmentation
from training.lora_adapter import apply_lora_to_backbone, save_lora_for_region

log = structlog.get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="base", choices=["nano", "tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "models" / "olmoearth_cocoa_seg_base.pt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--region", default="ghana")
    parser.add_argument("--lora", action="store_true", help="Train PEFT LoRA adapter + cocoa head")
    args = parser.parse_args(argv)
    model = OlmoEarthCocoaSegmentation(model_size=args.model_size, use_hf=False).to(args.device)
    if args.lora:
        model.backbone = apply_lora_to_backbone(model.backbone, "olmoearth")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        batch = {
            "s2": torch.randn(2, 4, 64, 64, 10, device=args.device),
            "s1": torch.randn(2, 4, 64, 64, 2, device=args.device),
            "era5": torch.randn(2, 4, 5, device=args.device),
            "dem": torch.randn(2, 64, 64, 2, device=args.device),
            "location": torch.zeros(2, 2, device=args.device),
            "months": torch.tensor([[6, 7, 8, 9], [6, 7, 8, 9]], device=args.device),
        }
        logits = model(batch)
        target = torch.zeros_like(logits)
        loss = loss_fn(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    if args.lora:
        adapter_path = save_lora_for_region(
            model.backbone,
            args.region,
            Path("models"),
            backbone_name="olmoearth",
        )
        log.info("saved_olmoearth_lora_adapter", path=str(adapter_path))
    log.info("wrote_checkpoint", path=str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
