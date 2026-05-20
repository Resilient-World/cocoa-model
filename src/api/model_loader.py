"""Load the yield surrogate checkpoint for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.checkpoint_migration import is_v1_static_checkpoint, migrate_v1_static_to_v2
from models.yield_surrogate import YieldSurrogateModel

if TYPE_CHECKING:
    from api.config import APISettings

logger = logging.getLogger(__name__)


def load_yield_model(
    checkpoint_path: str | None = None,
    *,
    settings: APISettings | None = None,
) -> YieldSurrogateModel:
    """
    Instantiate :class:`YieldSurrogateModel` and optionally load trained weights.

    When ``settings.use_galileo_embedding`` is true, expands static input to
    ``13 + galileo_embedding_dim`` (Galileo tail concatenated by the feature resolver).

    Legacy v1 checkpoints (10 site static features) are auto-migrated via
    :func:`~models.checkpoint_migration.migrate_v1_static_to_v2`.
    """
    galileo_dim = 0
    if settings is not None and settings.use_galileo_embedding:
        galileo_dim = settings.galileo_embedding_dim

    model = YieldSurrogateModel(galileo_dim=galileo_dim)
    path = Path(checkpoint_path) if checkpoint_path else None

    if path is not None and path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and is_v1_static_checkpoint(state):
            state = migrate_v1_static_to_v2(state)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded yield model weights from %s", path)
    else:
        logger.warning(
            "No checkpoint at %s; using uninitialized weights for inference",
            checkpoint_path,
        )

    model.eval()
    return model
