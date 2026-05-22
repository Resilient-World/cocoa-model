"""Tests for Galileo backbone and segmentation heads."""

from __future__ import annotations

import pytest
import torch

from models.galileo_backbone import GalileoCocoaBackbone, GalileoSegmentation


@pytest.mark.slow
def test_galileo_nano_forward_smoke() -> None:
    m = GalileoCocoaBackbone(model_size="nano", freeze=True)
    s2 = torch.randn(1, 2, 8, 8, 10)  # B,T,H,W,bands
    out = m({"s2": s2})
    assert out.dim() == 4 and out.shape[0] == 1


@pytest.mark.slow
def test_galileo_segmentation_output_shape() -> None:
    m = GalileoSegmentation(model_size="nano", num_classes=3)
    s2 = torch.randn(1, 2, 64, 64, 10)
    logits = m({"s2": s2})
    assert logits.shape == (1, 3, 64, 64)


def test_freeze_flag_disables_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze is enforced on encoder parameters without downloading HF weights."""
    m = GalileoCocoaBackbone(model_size="nano", freeze=True)
    stub = torch.nn.Linear(4, 4)
    for param in stub.parameters():
        param.requires_grad = False
    monkeypatch.setattr(m, "_encoder", stub)
    m._embedding_size = 4
    assert all(not p.requires_grad for p in m.encoder.parameters())
