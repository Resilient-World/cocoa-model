"""Load the yield surrogate checkpoint for inference."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from models.yield_surrogate import YieldSurrogateModel

logger = logging.getLogger(__name__)


def load_yield_model(checkpoint_path: str | None = None) -> YieldSurrogateModel:
    """
    Instantiate :class:`YieldSurrogateModel` and optionally load trained weights.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` / ``.pth`` state dict. If missing or unset, returns
        an uninitialized model (suitable for mocked demo inference).
    """
    model = YieldSurrogateModel()
    path = Path(checkpoint_path) if checkpoint_path else None

    if path is not None and path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded yield model weights from %s", path)
    else:
        logger.warning(
            "No checkpoint at %s; using uninitialized weights for inference",
            checkpoint_path,
        )

    model.eval()
    return model
