"""FixMatch-style pseudo-label bootstrap for cocoa exposure refinement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

DEFAULT_PSEUDO_THRESHOLD = 0.95


class ProbabilisticModel(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class PseudoLabelBatch:
    pseudo_labels: Tensor
    confidence: Tensor
    mask: Tensor


def pseudo_threshold(default: float = DEFAULT_PSEUDO_THRESHOLD) -> float:
    """Read the FixMatch confidence threshold from ``PSEUDO_THRESHOLD``."""
    raw = os.environ.get("PSEUDO_THRESHOLD")
    if raw is None:
        return default
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError("PSEUDO_THRESHOLD must be between 0 and 1")
    return value


def random_crop_and_flip(x: Tensor, *, crop_scale: float = 0.875) -> Tensor:
    """Weak augmentation: random crop resized to input shape plus horizontal flip."""
    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    _, _, h, w = x.shape
    crop_h = max(1, int(h * crop_scale))
    crop_w = max(1, int(w * crop_scale))
    top = torch.randint(0, h - crop_h + 1, (1,), device=x.device).item()
    left = torch.randint(0, w - crop_w + 1, (1,), device=x.device).item()
    out = x[:, :, top : top + crop_h, left : left + crop_w]
    out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    if bool(torch.randint(0, 2, (1,), device=x.device).item()):
        out = torch.flip(out, dims=(-1,))
    return out


def randaugment_sentinel_noise(
    x: Tensor,
    *,
    magnitude: float = 0.08,
    band_noise_std: float = 0.02,
) -> Tensor:
    """Strong augmentation: RandAugment-like affine jitter plus Sentinel-2 band noise."""
    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    out = x
    if bool(torch.randint(0, 2, (1,), device=x.device).item()):
        out = torch.flip(out, dims=(-1,))
    brightness = 1.0 + (torch.rand(1, device=x.device).item() * 2.0 - 1.0) * magnitude
    contrast = 1.0 + (torch.rand(1, device=x.device).item() * 2.0 - 1.0) * magnitude
    channel_mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - channel_mean) * contrast + channel_mean
    out = out * brightness
    noise = torch.randn_like(out) * band_noise_std
    return out + noise


def weak_augment(x: Tensor) -> Tensor:
    return random_crop_and_flip(x)


def strong_augment(x: Tensor) -> Tensor:
    return randaugment_sentinel_noise(random_crop_and_flip(x))


@torch.no_grad()
def make_pseudo_labels(
    logits_or_probabilities: Tensor,
    *,
    threshold: float | None = None,
) -> PseudoLabelBatch:
    """Gate high-confidence pseudo-labels for binary or multiclass predictions."""
    thresh = pseudo_threshold() if threshold is None else threshold
    preds = logits_or_probabilities
    if preds.ndim == 1 or preds.shape[-1] == 1:
        probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1) else preds
        confidence = torch.maximum(probs, 1.0 - probs)
        labels = (probs >= 0.5).long()
    else:
        probs = torch.softmax(preds, dim=-1) if (preds.min() < 0 or preds.max() > 1) else preds
        confidence, labels = probs.max(dim=-1)
    mask = confidence >= thresh
    return PseudoLabelBatch(pseudo_labels=labels, confidence=confidence, mask=mask)


def fixmatch_consistency_loss(
    model: ProbabilisticModel,
    unlabeled_batch: Tensor,
    *,
    threshold: float | None = None,
) -> Tensor:
    """Weak-strong consistency loss with confidence-gated pseudo-labels."""
    weak = weak_augment(unlabeled_batch)
    strong = strong_augment(unlabeled_batch)
    with torch.no_grad():
        pseudo = make_pseudo_labels(model(weak), threshold=threshold)
    if not bool(pseudo.mask.any()):
        return strong.sum() * 0.0
    strong_logits = model(strong)
    if strong_logits.ndim == 1 or strong_logits.shape[-1] == 1:
        target = pseudo.pseudo_labels.float().view_as(strong_logits)
        loss = F.binary_cross_entropy_with_logits(strong_logits, target, reduction="none")
        return loss[pseudo.mask.view_as(loss)].mean()
    return F.cross_entropy(strong_logits[pseudo.mask], pseudo.pseudo_labels[pseudo.mask])


class PseudoLabelDataset(Dataset[tuple[Tensor, Tensor]]):
    """Tensor dataset wrapper for high-confidence pseudo-labels."""

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


@torch.no_grad()
def build_pseudo_label_dataset(
    model: ProbabilisticModel,
    unlabeled_loader: DataLoader[Tensor],
    *,
    threshold: float | None = None,
    device: str = "cpu",
) -> PseudoLabelDataset:
    """Run a pseudo-label pass and retain high-confidence samples."""
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    for batch in unlabeled_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        pseudo = make_pseudo_labels(model(weak_augment(x)), threshold=threshold)
        if bool(pseudo.mask.any()):
            xs.append(x[pseudo.mask].cpu())
            ys.append(pseudo.pseudo_labels[pseudo.mask].cpu())
    if not xs:
        return PseudoLabelDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    return PseudoLabelDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0))


def combine_labeled_and_pseudo(
    labeled_dataset: Dataset,
    pseudo_dataset: Dataset,
) -> ConcatDataset:
    """Combine human labels and gated pseudo-labels for self-training."""
    return ConcatDataset([labeled_dataset, pseudo_dataset])


def should_run_self_training(iteration: int, *, every_k: int) -> bool:
    """Trigger self-training every K active-learning iterations."""
    if every_k <= 0:
        return False
    return iteration > 0 and iteration % every_k == 0


__all__ = [
    "DEFAULT_PSEUDO_THRESHOLD",
    "PseudoLabelBatch",
    "PseudoLabelDataset",
    "build_pseudo_label_dataset",
    "combine_labeled_and_pseudo",
    "fixmatch_consistency_loss",
    "make_pseudo_labels",
    "pseudo_threshold",
    "randaugment_sentinel_noise",
    "random_crop_and_flip",
    "should_run_self_training",
    "strong_augment",
    "weak_augment",
]
