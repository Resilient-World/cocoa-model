"""Fine-tune OlmoEarth cocoa segmentation head on Kalischek tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.olmoearth_seg import OlmoEarthCocoaSegmentation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="base", choices=["nano", "tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "olmoearth_cocoa_seg_base.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    model = OlmoEarthCocoaSegmentation(model_size=args.model_size, use_hf=False).to(args.device)
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
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
