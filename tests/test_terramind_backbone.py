"""Tests for TerraMind backbone (proxy encoder in CPU CI)."""

from __future__ import annotations


def test_terramind_forward_smoke() -> None:
    import torch

    from data.utils import cocoa_batch_to_terramind_input
    from models.terramind_backbone import TerraMindBackbone

    model = TerraMindBackbone(freeze=True, pretrained=False)
    batch = {
        "s2": torch.randn(1, 12, 64, 64),
        "s1": torch.randn(1, 2, 64, 64),
        "dem": torch.randn(1, 2, 64, 64),
    }
    x = cocoa_batch_to_terramind_input(batch)
    out = model(x)
    assert out.dim() == 4
    assert out.shape[0] == 1
    assert out.shape[1] > 0


def test_terramind_freeze_disables_grad() -> None:
    import torch

    from models.terramind_backbone import TerraMindBackbone

    model = TerraMindBackbone(freeze=True, pretrained=False)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    model.set_freeze(False)
    x = {
        "S2L2A": torch.randn(1, 12, 32, 32),
        "S1GRD": torch.randn(1, 2, 32, 32),
        "DEM": torch.randn(1, 2, 32, 32),
    }
    model(x).sum().backward()
    assert any(p.requires_grad and p.grad is not None for p in model.encoder.parameters())


def test_cocoa_batch_to_terramind_input_keys() -> None:
    import torch

    from data.utils import cocoa_batch_to_terramind_input

    batch = {
        "s2": torch.randn(1, 12, 8, 8),
        "s1": torch.randn(1, 2, 8, 8),
        "dem": torch.randn(1, 2, 8, 8),
    }
    out = cocoa_batch_to_terramind_input(batch)
    assert "S2L2A" in out
    assert "S1GRD" in out
    assert "DEM" in out
