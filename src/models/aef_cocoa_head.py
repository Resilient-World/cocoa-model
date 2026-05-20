"""
Lightweight cocoa classifier on AlphaEarth Foundations 64-D embeddings.

Trained on Kalischek et al. (2023) in-situ labels with frozen AEF vectors as input
(near-zero inference cost — embeddings are pre-computed on Earth Engine).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.alphaearth_embeddings import AEF_EMBEDDING_DIM

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AEF_CHECKPOINT = _REPO_ROOT / "models" / "aef_cocoa_head.pt"


class AEFCocoaHead(nn.Module):
    """
    Two-layer MLP: 64 → 128 → 1 with sigmoid for P(cocoa).

    Parameters
    ----------
    embedding_dim:
        Input width (default 64 AlphaEarth bands).
    hidden_dim:
        Hidden layer width.
    dropout:
        Dropout probability during training.
    """

    def __init__(
        self,
        embedding_dim: int = AEF_EMBEDDING_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings:
            ``[B, 64]`` or ``[B, 1, 64]`` unit-norm AEF vectors.

        Returns
        -------
        torch.Tensor
            Logits ``[B]`` (apply sigmoid for probability).
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected last dim {self.embedding_dim}, got {embeddings.shape[-1]}"
            )
        return self.net(embeddings).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """P(cocoa) in [0, 1], shape ``[B]``."""
        self.eval()
        return torch.sigmoid(self.forward(embeddings))

    def bce_loss(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        *,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        """Binary cross-entropy against soft or hard targets in [0, 1]."""
        logits = self.forward(embeddings)
        weight = None
        if pos_weight is not None:
            weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.view_as(logits).clamp(0.0, 1.0),
            pos_weight=weight,
        )


def load_aef_cocoa_head(
    path: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
) -> AEFCocoaHead:
    """Load trained head weights from ``aef_cocoa_head.pt``."""
    checkpoint = Path(path) if path else DEFAULT_AEF_CHECKPOINT
    model = AEFCocoaHead()
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded AEFCocoaHead from %s", checkpoint)
    else:
        logger.warning("AEF head checkpoint missing at %s; using random weights", checkpoint)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "AEFCocoaHead",
    "DEFAULT_AEF_CHECKPOINT",
    "load_aef_cocoa_head",
]
