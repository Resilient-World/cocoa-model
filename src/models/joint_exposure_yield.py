"""
Joint cocoa exposure (segmentation) and yield head on shared backbone features.

Prithvi / Galileo / AlphaEarth embeddings already encode yield-relevant structure;
this module shares a single feature map for P(cocoa) and tonnes/ha (with CQR quantiles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.cqr import DEFAULT_QUANTILES, pinball_loss

DEFAULT_LAMBDA_CQR = 0.1


class JointOutputs(NamedTuple):
    """Forward pass outputs."""

    seg_logits: Tensor  # [B, 1, H, W]
    yield_point: Tensor  # [B]
    yield_quantiles: Tensor  # [B, Q]


@dataclass
class JointLossBreakdown:
    """Scalar loss components for logging."""

    total: float
    seg_bce: float
    yield_mse: float
    yield_pinball: float


class JointHead(nn.Module):
    """
    Multi-task head on dense backbone features.

    Branch A: 1×1 conv → cocoa probability (segmentation logits).
    Branch B: global average pool → concat static site vector → MLP → yield + quantiles.
    """

    def __init__(
        self,
        backbone_dim: int,
        static_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        if backbone_dim <= 0:
            raise ValueError(f"backbone_dim must be positive, got {backbone_dim}")
        if static_dim <= 0:
            raise ValueError(f"static_dim must be positive, got {static_dim}")

        self.backbone_dim = backbone_dim
        self.static_dim = static_dim
        self.quantiles = tuple(float(q) for q in quantiles)
        self.n_quantiles = len(self.quantiles)

        self.seg_conv = nn.Conv2d(backbone_dim, 1, kernel_size=1)
        mlp_in = backbone_dim + static_dim
        self.yield_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1 + self.n_quantiles),
        )

    def forward(
        self,
        backbone_features: Tensor,
        static_features: Tensor,
    ) -> JointOutputs:
        """
        Parameters
        ----------
        backbone_features:
            ``[B, D, H, W]`` dense map (e.g. collapsed Galileo space-time tokens).
        static_features:
            ``[B, S]`` site static vector aligned with yield surrogate.
        """
        if backbone_features.dim() != 4:
            raise ValueError(
                f"backbone_features must be [B, D, H, W], got {tuple(backbone_features.shape)}"
            )
        if static_features.dim() != 2:
            raise ValueError(
                f"static_features must be [B, S], got {tuple(static_features.shape)}"
            )
        if backbone_features.shape[1] != self.backbone_dim:
            raise ValueError(
                f"Expected backbone_dim={self.backbone_dim}, got {backbone_features.shape[1]}"
            )
        if static_features.shape[1] != self.static_dim:
            raise ValueError(
                f"Expected static_dim={self.static_dim}, got {static_features.shape[1]}"
            )

        seg_logits = self.seg_conv(backbone_features)
        pooled = backbone_features.mean(dim=(2, 3))
        fused = torch.cat([pooled, static_features], dim=-1)
        yield_out = self.yield_mlp(fused)
        yield_point = yield_out[:, 0]
        yield_quantiles = yield_out[:, 1:]
        return JointOutputs(seg_logits, yield_point, yield_quantiles)

    @staticmethod
    def global_pool_backbone(backbone_features: Tensor) -> Tensor:
        """``[B, D, H, W]`` → ``[B, D]``."""
        return backbone_features.mean(dim=(2, 3))


class JointMultiTaskLoss(nn.Module):
    """
    ``L = BCE(seg) + MSE(yield) + λ · mean_pinball(quantiles)``.

    Segmentation targets are soft or hard masks in ``[0, 1]``. Yield MSE is applied to
    the median quantile (index 1 when ``quantiles=(0.05, 0.5, 0.95)``).
    """

    def __init__(
        self,
        *,
        lambda_cqr: float = DEFAULT_LAMBDA_CQR,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_cqr = float(lambda_cqr)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.register_buffer(
            "_pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(
        self,
        outputs: JointOutputs,
        *,
        seg_target: Tensor,
        yield_target: Tensor,
    ) -> tuple[Tensor, JointLossBreakdown]:
        """
        Parameters
        ----------
        seg_target:
            ``[B, 1, H, W]`` or ``[B, H, W]`` binary/soft cocoa mask.
        yield_target:
            ``[B]`` yield in tonnes/ha.
        """
        logits = outputs.seg_logits
        if seg_target.dim() == 3:
            seg_target = seg_target.unsqueeze(1)
        if seg_target.shape[-2:] != logits.shape[-2:]:
            seg_target = F.interpolate(
                seg_target.float(),
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pw = self._pos_weight if self._pos_weight is not None else None
        seg_bce = F.binary_cross_entropy_with_logits(
            logits,
            seg_target.clamp(0.0, 1.0),
            pos_weight=pw,
        )

        q = outputs.yield_quantiles
        if q.shape[1] != len(self.quantiles):
            raise ValueError(f"Expected {len(self.quantiles)} quantiles, got {q.shape[1]}")

        median_idx = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        yield_mse = F.mse_loss(q[:, median_idx], yield_target.view_as(q[:, median_idx]))
        yield_pinball = pinball_loss(q, yield_target, quantiles=self.quantiles)

        total = seg_bce + yield_mse + self.lambda_cqr * yield_pinball
        breakdown = JointLossBreakdown(
            total=float(total.detach()),
            seg_bce=float(seg_bce.detach()),
            yield_mse=float(yield_mse.detach()),
            yield_pinball=float(yield_pinball.detach()),
        )
        return total, breakdown


def load_joint_head(
    path: str | None = None,
    *,
    backbone_dim: int = 128,
    static_dim: int = 13,
    device: str | torch.device = "cpu",
) -> JointHead:
    """Load :class:`JointHead` weights from ``models/joint.pt``."""
    from pathlib import Path

    checkpoint = Path(path) if path else Path(__file__).resolve().parents[2] / "models" / "joint.pt"
    model = JointHead(backbone_dim=backbone_dim, static_dim=static_dim)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            cfg = state.get("config", {})
            backbone_dim = int(cfg.get("backbone_dim", backbone_dim))
            static_dim = int(cfg.get("static_dim", static_dim))
            model = JointHead(backbone_dim=backbone_dim, static_dim=static_dim)
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "DEFAULT_LAMBDA_CQR",
    "JointHead",
    "JointLossBreakdown",
    "JointMultiTaskLoss",
    "JointOutputs",
    "load_joint_head",
]
