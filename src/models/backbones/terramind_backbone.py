"""
TerraMind 1.0 encoder backbone (IBM-ESA / TerraTorch).

Loads ``terramind_v1_base`` from ``terratorch.registry.TERRATORCH_BACKBONE_REGISTRY``
when ``pip install -e ".[terramind]"`` is satisfied. Falls back to a lightweight
conv proxy for CPU CI when the registry is unavailable.
"""

from __future__ import annotations

import structlog

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

log = structlog.get_logger(__name__)

TERRAMIND_HF_REPO = "ibm-esa-geospatial/TerraMind-1.0-base"
DEFAULT_VARIANT = "terramind_v1_base"
DEFAULT_TIM_VARIANT = "terramind_v1_base_tim"

ModalityKey = Literal["S2L2A", "S1GRD", "DEM"]
DEFAULT_MODALITIES: tuple[ModalityKey, ...] = ("S2L2A", "S1GRD", "DEM")

S2L2A_CHANNELS = 12
S1GRD_CHANNELS = 2
DEM_CHANNELS = 2


def require_terramind_extra() -> None:
    """Raise if the optional ``[terramind]`` dependencies are not installed."""
    import importlib.util

    if importlib.util.find_spec("terratorch") is None:
        raise ImportError(
            'TerraMind requires optional deps: pip install -e ".[terramind]"'
        )


def _collapse_tokens_to_map(tensor: torch.Tensor) -> torch.Tensor:
    """
    Collapse TerraMind token layouts to ``[B, D, H, W]``.

    Supports ``[B, D, H, W]``, ``[B, N, D]`` (square grid), or ``[B, T, H, W, D]``.
    """
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 5:
        # [B, T, H, W, D] -> mean over time, permute to BCHW
        return tensor.mean(dim=1).permute(0, 3, 1, 2).contiguous()
    if tensor.dim() == 3:
        b, n, d = tensor.shape
        side = int(n**0.5)
        if side * side == n:
            return tensor.transpose(1, 2).reshape(b, d, side, side)
        return tensor.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(b, d, 8, 8)
    raise ValueError(f"Unsupported TerraMind feature shape {tuple(tensor.shape)}")


class _ProxyTerraMindEncoder(nn.Module):
    """MIT reimplementation stub when TerraTorch TerraMind registry is unavailable."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._stem: nn.Sequential | None = None
        self._stem_in_ch: int | None = None

    def _stem_for(self, in_ch: int) -> nn.Sequential:
        if self._stem is not None and self._stem_in_ch == in_ch:
            return self._stem
        self._stem_in_ch = in_ch
        self._stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )
        return self._stem

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        target_hw: tuple[int, int] | None = None
        for key in list(x.keys()):
            t = x[key]
            if t.dim() == 5:
                t = t.mean(dim=1)
            if t.dim() == 3:
                t = t.unsqueeze(1)
            if target_hw is None:
                target_hw = (t.shape[-2], t.shape[-1])
            elif t.shape[-2:] != target_hw:
                t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
            tensors.append(t)
        if not tensors:
            raise ValueError("TerraMind input dict is empty")
        cat = torch.cat(tensors, dim=1)
        return self._stem_for(cat.shape[1])(cat)


def _load_terramind_encoder(
    variant: str = DEFAULT_VARIANT,
    *,
    pretrained: bool = False,
    tim: bool = False,
) -> nn.Module:
    """Instantiate TerraMind from TerraTorch registry or proxy fallback."""
    try:
        require_terramind_extra()
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

        key = DEFAULT_TIM_VARIANT if tim else variant
        if key not in TERRATORCH_BACKBONE_REGISTRY:
            log.warning("TerraMind variant %s not in registry; using proxy encoder", key)
            return _ProxyTerraMindEncoder()
        factory = TERRATORCH_BACKBONE_REGISTRY.get(key)
        model = factory(pretrained=pretrained, modalities=list(DEFAULT_MODALITIES))
        return model
    except Exception as exc:
        log.warning("TerraMind registry load failed (%s); using proxy encoder", exc)
        return _ProxyTerraMindEncoder()


class TerraMindBackbone(nn.Module):
    """
    TerraMind 1.0 multi-modal encoder producing a dense feature map.

    Parameters
    ----------
    variant:
        TerraTorch registry key (default ``terramind_v1_base``).
    freeze:
        Freeze encoder weights when True.
    modalities:
        Expected input keys in :meth:`forward`.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        freeze: bool = True,
        *,
        modalities: tuple[ModalityKey, ...] = DEFAULT_MODALITIES,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.modalities = modalities
        self.encoder = _load_terramind_encoder(variant, pretrained=pretrained, tim=False)
        self._is_proxy = isinstance(self.encoder, _ProxyTerraMindEncoder)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def set_freeze(self, freeze: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def _run_encoder(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._is_proxy:
            return self.encoder(x)
        out = self.encoder(x)
        if isinstance(out, dict):
            feat = out.get("features") or out.get("last_hidden_state") or next(iter(out.values()))
        elif isinstance(out, (list, tuple)):
            feat = out[-1]
        else:
            feat = out
        return _collapse_tokens_to_map(feat)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Dict with ``S2L2A``, ``S1GRD``, ``DEM`` tensors ``[B, C, H, W]`` (or 5D with time).

        Returns
        -------
        torch.Tensor
            Feature map ``[B, D, H', W']``.
        """
        subset = {k: x[k] for k in self.modalities if k in x}
        if not subset:
            raise KeyError(f"TerraMind forward expects one of {self.modalities}; got {list(x)}")
        return self._run_encoder(subset)


def load_terramind_backbone(
    *,
    variant: str = DEFAULT_VARIANT,
    freeze: bool = True,
    pretrained: bool = False,
) -> TerraMindBackbone:
    """Construct :class:`TerraMindBackbone` (registry or proxy)."""
    return TerraMindBackbone(variant=variant, freeze=freeze, pretrained=pretrained)


__all__ = [
    "DEFAULT_MODALITIES",
    "DEFAULT_TIM_VARIANT",
    "DEFAULT_VARIANT",
    "DEM_CHANNELS",
    "ModalityKey",
    "S1GRD_CHANNELS",
    "S2L2A_CHANNELS",
    "TERRAMIND_HF_REPO",
    "TerraMindBackbone",
    "load_terramind_backbone",
    "require_terramind_extra",
]
