"""Tests for AgriFM cocoa fine-tuning pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_agrifm_cocoa.py"


@pytest.mark.slow
def test_synthetic_training_exports_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "agrifm_cocoa_seg.pt"
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--synthetic",
        "--max-tiles",
        "100",
        "--epochs",
        "1",
        "--quick",
        "--out",
        str(out),
        "--accelerator",
        "cpu",
        "--devices",
        "1",
    ]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "state_dict" in payload


def test_agrifm_bce_dice_loss() -> None:
    from training.agrifm_losses import agrifm_bce_dice_loss

    logits = torch.randn(2, 1, 16, 16)
    target = torch.zeros(2, 1, 16, 16)
    target[0, :, :8] = 1.0
    loss = agrifm_bce_dice_loss(logits, target, pos_weight=4.0)
    assert loss.ndim == 0
    assert float(loss) > 0.0
