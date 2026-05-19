from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .vendor.single_file_galileo import Encoder as GalileoEncoder

GALILEO_HF_REPO = "nasaharvest/galileo"
SUPPORTED_SIZES = {"nano", "tiny", "base"}


def download_galileo_weights(size: str = "base", cache_dir: str | Path | None = None) -> Path:
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {size}")
    local = snapshot_download(
        repo_id=GALILEO_HF_REPO,
        allow_patterns=[f"models/{size}/*"],
        cache_dir=cache_dir,
    )
    return Path(local) / "models" / size


def load_galileo(
    size: str = "base",
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
) -> GalileoEncoder:
    weights_dir = download_galileo_weights(size, cache_dir=cache_dir)
    model = GalileoEncoder.load_from_folder(weights_dir, device=torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
