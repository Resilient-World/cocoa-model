"""Tests for AgriFM Video Swin backbone and segmentation head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.agrifm_backbone import MAX_FRAMES, MIN_FRAMES, AgriFMBackbone
from models.agrifm_cocoa_head import AgriFMCocoaSegHead
from models.agrifm_seg import AgriFMCocoaSegmentation


def _random_backbone(
    *,
    freeze: bool = True,
    checkpoint: Path | None = None,
    num_frames: int | None = None,
) -> AgriFMBackbone:
    ckpt = checkpoint or Path("/nonexistent/agrifm.pt")
    return AgriFMBackbone(
        checkpoint_path=ckpt,
        modality="S2",
        freeze=freeze,
        num_frames=num_frames,
    )


@pytest.mark.slow
def test_forward_smoke_bthwc_stages() -> None:
    model = _random_backbone()
    x = torch.randn(2, 10, 8, 224, 224)
    out = model(x)
    assert set(out.keys()) == {"stage_1", "stage_2", "stage_3", "stage_4"}
    h_prev, w_prev = 224, 224
    for key in ("stage_1", "stage_2", "stage_3", "stage_4"):
        t = out[key]
        assert t.shape[0] == 2
        assert t.dim() == 5
        assert t.shape[-1] > 0
        assert t.shape[2] <= h_prev
        assert t.shape[3] <= w_prev
        h_prev, w_prev = t.shape[2], t.shape[3]


@pytest.mark.parametrize("num_frames", [3, 16, 32])
def test_variable_temporal_length(num_frames: int) -> None:
    model = _random_backbone(num_frames=num_frames)
    x = torch.randn(1, 10, num_frames, 112, 112)
    out = model(x)
    for stage in out.values():
        assert stage.shape[0] == 1
        assert stage.shape[1] >= 1


def test_freeze_disables_grad() -> None:
    model = _random_backbone(freeze=True)
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_resolve_num_frames_bounds() -> None:
    model = _random_backbone()
    with pytest.raises(ValueError):
        model.resolve_num_frames(MIN_FRAMES - 1)
    with pytest.raises(ValueError):
        model.resolve_num_frames(MAX_FRAMES + 1)


def test_checkpoint_loading_mocked(tmp_path: Path) -> None:
    model = _random_backbone(freeze=False)
    state = {
        "S2_patch_emd.proj.weight": model.encoder.patch_embed.proj.weight.clone(),
        "S2_patch_emd.proj.bias": model.encoder.patch_embed.proj.bias.clone(),
    }
    ckpt_path = tmp_path / "mock_agrifm.pt"
    torch.save(state, ckpt_path)
    loaded = AgriFMBackbone(checkpoint_path=ckpt_path, modality="S2", freeze=False)
    assert (
        loaded.encoder.patch_embed.proj.weight.shape == model.encoder.patch_embed.proj.weight.shape
    )


def test_cocoa_head_output_shape() -> None:
    head = AgriFMCocoaSegHead(out_size=(64, 64))
    stages = {
        "stage_1": torch.randn(1, 2, 56, 56, 128),
        "stage_2": torch.randn(1, 1, 28, 28, 256),
        "stage_3": torch.randn(1, 1, 14, 14, 512),
        "stage_4": torch.randn(1, 1, 7, 7, 1024),
    }
    logits = head(stages)
    assert logits.shape == (1, 1, 64, 64)


def test_segmentation_predict_proba_numpy() -> None:
    seg = AgriFMCocoaSegmentation(
        checkpoint_path=Path("/nonexistent/agrifm.pt"),
        out_size=(32, 32),
        num_frames=4,
    )
    s2 = torch.randn(1, 4, 32, 32, 10)
    prob = seg.predict_proba_numpy(s2)
    assert prob.shape == (32, 32)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
