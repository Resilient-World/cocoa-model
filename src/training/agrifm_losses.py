"""Binary segmentation losses for AgriFM cocoa fine-tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss on sigmoid probabilities.

    Parameters
    ----------
    logits:
        ``[B, 1, H, W]`` raw logits.
    target:
        ``[B, 1, H, W]`` binary targets in ``{0, 1}``.
    """
    prob = torch.sigmoid(logits)
    target = target.float()
    if target.dim() == 3:
        target = target.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def agrifm_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float = 4.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE + Dice for imbalanced cocoa pixels.

    Parameters
    ----------
    logits:
        ``[B, 1, H, W]``.
    target:
        ``[B, 1, H, W]`` or ``[B, H, W]`` in ``[0, 1]``.
    pos_weight:
        Positive class weight (default 4.0 per plan).
    dice_weight:
        Scale on Dice term.
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target.clamp(0.0, 1.0),
        pos_weight=pw,
    )
    dice = dice_loss(logits, target)
    return bce + dice_weight * dice
