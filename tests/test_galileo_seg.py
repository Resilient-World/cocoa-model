"""Tests for Galileo binary cocoa segmentation head."""

from __future__ import annotations

import torch
import pytest

from models.galileo_seg import ERA5_GALILEO_COUNT, GalileoCocoaSegmentation


def test_build_batch_dict_era5_maps_to_galileo_channels() -> None:
    t, h, w = 4, 16, 16
    batch = GalileoCocoaSegmentation.build_batch_dict(
        s2=torch.randn(1, t, h, w, 10),
        s1=torch.randn(1, t, h, w, 2),
        era5=torch.randn(1, t, 5),
        dem=torch.randn(1, h, w, 2),
    )
    assert batch["era5"].shape == (t, ERA5_GALILEO_COUNT)
    assert batch["s2"].shape == (t, h, w, 10)


def test_weak_strong_bce_loss_combines_targets() -> None:
    logits = torch.randn(1, 1, 8, 8)
    fdp = torch.rand(1, 1, 8, 8)
    kal = torch.randint(0, 2, (1, 1, 8, 8)).float()
    model = GalileoCocoaSegmentation(model_size="nano", freeze_backbone=True)
    loss = model.weak_strong_bce_loss(logits, fdp_target=fdp, kalischek_target=kal)
    assert loss.ndim == 0 and torch.isfinite(loss)


@pytest.mark.slow
def test_galileo_seg_forward_smoke() -> None:
    m = GalileoCocoaSegmentation(model_size="nano", freeze_backbone=True)
    t, h, w = 2, 32, 32
    batch = m.build_batch_dict(
        s2=torch.randn(t, h, w, 10),
        s1=torch.randn(t, h, w, 2),
        era5=torch.randn(t, 2),
        dem=torch.randn(h, w, 2),
    )
    logits = m(batch)
    assert logits.shape == (1, 1, h, w)
