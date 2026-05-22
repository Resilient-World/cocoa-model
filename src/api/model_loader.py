"""Load the yield surrogate checkpoint for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
import torch

from models.casej_surrogate import CASEJSurrogate, load_casej_surrogate
from models.checkpoint_migration import is_v1_static_checkpoint, migrate_v1_static_to_v2
from models.yield_surrogate import YieldSurrogateModel
from models.yield_surrogate_v2 import YieldSurrogateV2
from models.yield_surrogate_v2_teleconnection import YieldSurrogateV2Teleconnection

if TYPE_CHECKING:
    from api.config import APISettings

log = structlog.get_logger(__name__)

_V1_DEFAULT = "models/yield_surrogate_v1.pt"
_V2_DEFAULT = "models/yield_surrogate_v2.pt"
_TELECONNECTION_DEFAULT = "models/yield_surrogate_v2_teleconnection.pt"

YieldModel = YieldSurrogateModel | YieldSurrogateV2 | YieldSurrogateV2Teleconnection


@runtime_checkable
class YieldSurrogateProtocol(Protocol):
    """Common inference surface for yield surrogates."""

    def forward(
        self,
        climate: torch.Tensor,
        static: torch.Tensor,
        region_id: torch.Tensor | None = None,
        teleconnection: object | None = None,
        *,
        doy: torch.Tensor | None = None,
        lat: float | torch.Tensor = 6.0,
        lon: float | torch.Tensor = -2.0,
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


def _teleconnection_enabled(settings: APISettings | None) -> bool:
    if settings is None:
        return False
    if not settings.enable_teleconnection:
        return False
    return settings.teleconnection_checkpoint_path.is_file()


def _load_yield_from_registry(model_name: str) -> YieldModel | None:
    from api.telemetry import trace_span

    with trace_span("mlflow.registry.load", model_name=model_name):
        return _load_yield_from_registry_impl(model_name)


def _load_yield_from_registry_impl(model_name: str) -> YieldModel | None:
    try:
        from mlflow.exceptions import MlflowException

        from registry.mlflow_registry import get_champion
    except ImportError:
        return None
    try:
        pyfunc = get_champion(model_name)
    except MlflowException as exc:
        log.warning("mlflow_registry_miss", model=model_name, error=str(exc))
        return None
    impl = getattr(pyfunc, "_model_impl", None)
    inner = getattr(impl, "_model", None) if impl is not None else None
    if inner is not None:
        log.info("Loaded yield model from MLflow registry", model_name=model_name)
        inner.eval()
        return inner
    log.warning("mlflow_registry_unwrap_failed", model=model_name)
    return None


def load_yield_model(
    checkpoint_path: str | None = None,
    *,
    settings: APISettings | None = None,
) -> YieldModel:
    """
    Load yield surrogate (v1/v2) optionally wrapped with teleconnection GNN.

    When ``ENABLE_TELECONNECTION`` is true and the teleconnection checkpoint exists,
    returns :class:`~models.yield_surrogate_v2_teleconnection.YieldSurrogateV2Teleconnection`.
    """
    if settings is not None and settings.mlflow_registry_enabled:
        reg = _load_yield_from_registry(settings.mlflow_registry_model_name)
        if reg is not None:
            return reg

    if _teleconnection_enabled(settings):
        assert settings is not None
        sur_path = settings.model_checkpoint_path or _V2_DEFAULT
        if not Path(sur_path).is_file():
            sur_path = _V2_DEFAULT
        model = YieldSurrogateV2Teleconnection.from_checkpoints(
            sur_path,
            settings.teleconnection_checkpoint_path,
        )
        log.info(
            "Loaded YieldSurrogateV2 + teleconnection from %s",
            settings.teleconnection_checkpoint_path,
        )
        return model

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
            log.info("Loaded YieldSurrogateV2 from %s", v2_path)
            model.eval()
            return model
        except Exception as exc:
            log.warning("Failed to load v2 checkpoint %s (%s)", v2_path, exc)

    allow_fallback = settings is None or settings.allow_v1_fallback
    fallback = Path(_V1_DEFAULT) if Path(_V1_DEFAULT).is_file() else v2_path
    if allow_fallback and fallback is not None and fallback.is_file():
        log.warning("Using v1 weights via YieldSurrogateV2.from_v1_checkpoint (%s)", fallback)
        return YieldSurrogateV2.from_v1_checkpoint(fallback)

    log.warning("No checkpoint found; returning uninitialized YieldSurrogateV2")
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
        log.info("Loaded yield model v1 weights from %s", path)
    else:
        log.warning("No v1 checkpoint at %s; using uninitialized weights", path)
    model.eval()
    return model


def load_casej_model(
    checkpoint_path: str | None = None,
    *,
    settings: APISettings | None = None,
) -> CASEJSurrogate:
    """Load :class:`~models.casej_surrogate.CASEJSurrogate` (legacy scenario fallback)."""
    galileo_dim = 0
    if settings is not None and settings.use_galileo_embedding:
        galileo_dim = settings.galileo_embedding_dim
    return load_casej_surrogate(checkpoint_path, galileo_dim=galileo_dim)
