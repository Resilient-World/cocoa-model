"""Tests for PEFT LoRA adapter utilities."""

from __future__ import annotations

import torch
from torch import nn

from training.lora_adapter import (
    apply_lora_to_backbone,
    save_lora_for_region,
    trainable_parameter_fraction,
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.relu(self.q_proj(x) + self.v_proj(x)))


def test_apply_lora_preserves_shape_and_reduces_trainable_params(tmp_path) -> None:
    base = TinyBackbone()
    full_params = sum(p.numel() for p in base.parameters())
    model = apply_lora_to_backbone(
        base,
        "olmoearth",
        r=2,
        alpha=4,
        target_modules=("q_proj", "v_proj", "out_proj"),
    )
    out = model(torch.randn(3, 8))
    assert out.shape == (3, 8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < full_params
    assert trainable_parameter_fraction(model) < 0.75

    adapter = save_lora_for_region(model, "ghana", tmp_path, backbone_name="olmoearth")
    assert adapter.is_file()
    assert adapter.stat().st_size < 100_000
