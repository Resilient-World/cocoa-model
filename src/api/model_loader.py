"""Load the yield surrogate checkpoint for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from models.checkpoint_migration import is_v1_static_checkpoint, migrate_v1_static_to_v2
from models.casej_surrogate import CASEJSurrogate, load_casej_surrogate
from models.yield_surrogate import YieldSurrogateModel
from models.yield_surrogate_v2 import YieldSurrogateV2

if TYPE_CHECKING:
    from api.config import APISettings

logger = logging.getLogger(__name__)

_V1_DEFAULT = "models/yield_surrogate_v1.pt"
_V2_DEFAULT = "models/yield_surrogate_v2.pt"


@runtime_checkable
class YieldSurrogateProtocol(Protocol):
    """Common inference surface for v1 and v2 yield surrogates."""

    def forward(
        self,
        climate: torch.Tensor,
        static: torch.Tensor,
        region_id: torch.Tensor | None = None,
        *,
        doy: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def eval(self) -> YieldSurrogateProtocol: ...

    @property
    def training(self) -> bool: ...


def _resolve_checkpoint_path(
    checkpoint_path: str | None,
    *,
    settings: APISettings | None,
) -> tuple[Path | None, str]:
    version = settings.yield_surrogate_version if settings is not None else "v2"
    if checkpoint_path:
        return Path(checkpoint_path), version
    if settings is not None and settings.model_checkpoint_path:
        return Path(settings.model_checkpoint_path), version
    default = _V2_DEFAULT if version == "v2" else _V1_DEFAULT
    return Path(default), version


def load_yield_model(
    checkpoint_path: str | None = None,
    *,
    settings: APISettings | None = None,
) -> YieldSurrogateModel | YieldSurrogateV2:
    """
    Load :class:`YieldSurrogateModel` (v1) or :class:`YieldSurrogateV2` (v2).

    v2 uses PAPE with zero-init when loading v1 weights. Missing v2 checkpoints
    fall back to v1 via :meth:`YieldSurrogateV2.from_v1_checkpoint` when enabled.
    """
    path, version = _resolve_checkpoint_path(checkpoint_path, settings=settings)
    galileo_dim = 0
    if settings is not None and settings.use_galileo_embedding:
        galileo_dim = settings.galileo_embedding_dim

    if version == "v1":
        return _load_v1(path, galileo_dim=galileo_dim)

    v2_path = path

    if v2_path is not None and v2_path.is_file():
        try:
            model = YieldSurrogateV2.from_checkpoint(v2_path)
            logger.info("Loaded YieldSurrogateV2 from %s", v2_path)
            model.eval()
            return model
        except Exception as exc:
            logger.warning("Failed to load v2 checkpoint %s (%s)", v2_path, exc)

    allow_fallback = settings is None or settings.allow_v1_fallback
    fallback = Path(_V1_DEFAULT) if Path(_V1_DEFAULT).is_file() else v2_path
    if allow_fallback and fallback is not None and fallback.is_file():
        logger.warning("Using v1 weights via YieldSurrogateV2.from_v1_checkpoint (%s)", fallback)
        return YieldSurrogateV2.from_v1_checkpoint(fallback)

    logger.warning("No checkpoint found; returning uninitialized YieldSurrogateV2")
    model = YieldSurrogateV2(galileo_dim=galileo_dim)
    model.eval()
    return model


def _load_v1(path: Path | None, *, galileo_dim: int) -> YieldSurrogateModel:
    model = YieldSurrogateModel(galileo_dim=galileo_dim)
    if path is not None and path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and is_v1_static_checkpoint(state):
            state = migrate_v1_static_to_v2(state)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded yield model v1 weights from %s", path)
    else:
        logger.warning("No v1 checkpoint at %s; using uninitialized weights", path)
    model.eval()
    return model


def load_casej_model(
    checkpoint_path: str | None = None,
    *,
    settings: APISettings | None = None,
) -> CASEJSurrogate:
    """Load :class:`~models.casej_surrogate.CASEJSurrogate` for ``/simulate-scenario``."""
    galileo_dim = 0
    if settings is not None and settings.use_galileo_embedding:
        galileo_dim = settings.galileo_embedding_dim
    return load_casej_surrogate(checkpoint_path, galileo_dim=galileo_dim)
