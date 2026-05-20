"""Tests for :mod:`models.joint_exposure_yield`."""

from __future__ import annotations

import torch

from models.joint_exposure_yield import JointHead, JointMultiTaskLoss


def test_joint_head_forward_shapes() -> None:
    head = JointHead(backbone_dim=64, static_dim=13)
    b, h, w = 4, 8, 8
    backbone = torch.randn(b, 64, h, w)
    static = torch.randn(b, 13)
    out = head(backbone, static)
    assert out.seg_logits.shape == (b, 1, h, w)
    assert out.yield_point.shape == (b,)
    assert out.yield_quantiles.shape == (b, 3)


def test_joint_multitask_loss_decreases_with_good_fit() -> None:
    head = JointHead(backbone_dim=32, static_dim=8)
    loss_fn = JointMultiTaskLoss(lambda_cqr=0.1)
    b, h, w = 8, 4, 4
    backbone = torch.randn(b, 32, h, w)
    static = torch.randn(b, 8)
    seg_tgt = (backbone.mean(dim=1, keepdim=True) > 0).float()
    yield_tgt = torch.full((b,), 1.5)

    out = head(backbone, static)
    loss_bad, _ = loss_fn(out, seg_target=torch.zeros_like(seg_tgt), yield_target=yield_tgt)

    out_good = JointHead(backbone_dim=32, static_dim=8)
    out_good.seg_conv.weight.data = head.seg_conv.weight.data.clone()
    out_good.seg_conv.bias.data = head.seg_conv.bias.data.clone()
    out_good.yield_mlp.load_state_dict(head.yield_mlp.state_dict())
    pred = out_good(backbone, static)
    loss_good, bd = loss_fn(pred, seg_target=seg_tgt, yield_target=yield_tgt)
    assert loss_good.item() < loss_bad.item()
    assert bd.seg_bce >= 0.0
    assert bd.yield_mse >= 0.0
