"""
Galileo input utilities for cocoa ingest tensors.

Prefer the upstream ``galileo`` package (``pip install galileo @ git+...``); fall back to the
vendored copy under :mod:`models.vendor.galileo_data_utils` when the package is not installed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from models.vendor.galileo_data_utils import MaskedOutput

_construct_galileo_input: Callable[..., Any] | None = None
_MaskedOutput: type | None = None


def _resolve_galileo_utils() -> tuple[Callable[..., Any], type]:
    global _construct_galileo_input, _MaskedOutput
    if _construct_galileo_input is not None and _MaskedOutput is not None:
        return _construct_galileo_input, _MaskedOutput
    try:
        from galileo.data.utils import (  # type: ignore[import-untyped]
            MaskedOutput as _MO,
        )
        from galileo.data.utils import (  # type: ignore[import-untyped]
            construct_galileo_input as _cgi,
        )
    except ImportError:
        from models.vendor.galileo_data_utils import MaskedOutput as _MO
        from models.vendor.galileo_data_utils import construct_galileo_input as _cgi

    _construct_galileo_input = _cgi
    _MaskedOutput = _MO
    return _construct_galileo_input, _MaskedOutput


def construct_galileo_input(*args: Any, **kwargs: Any) -> MaskedOutput:
    """Build a Galileo :class:`MaskedOutput` (upstream or vendored implementation)."""
    cgi, _ = _resolve_galileo_utils()
    return cgi(*args, **kwargs)


def location_to_galileo_xyz(location: torch.Tensor) -> torch.Tensor:
    """
    Convert ``[lat, lon]`` (degrees) to Galileo location channels ``[x, y, z]``.

    Parameters
    ----------
    location:
        Tensor with last dimension 2 (lat, lon), any leading batch/spatial dims.
    """
    if location.shape[-1] != 2:
        raise ValueError(f"location must have last dim 2 (lat, lon); got shape {location.shape}")
    lat = location[..., 0]
    lon = location[..., 1]
    lat_rad = lat * math.pi / 180.0
    lon_rad = lon * math.pi / 180.0
    return torch.stack(
        [
            torch.cos(lat_rad) * torch.cos(lon_rad),
            torch.cos(lat_rad) * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ],
        dim=-1,
    )


def _thw_to_hwt(tensor: torch.Tensor) -> torch.Tensor:
    """``[T, H, W, C]`` or ``[B, T, H, W, C]`` → Galileo ``[H, W, T, C]`` layout."""
    if tensor.ndim == 4:
        return tensor.permute(1, 2, 0, 3)
    if tensor.ndim == 5:
        return tensor.permute(0, 2, 3, 1, 4)
    raise ValueError(f"Expected 4D or 5D (T,H,W,C); got {tensor.shape}")


def _maybe_hwt(
    tensor: torch.Tensor | None,
    *,
    expect_channels: int | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 4 and tensor.shape[1] > tensor.shape[-1]:
        # Heuristic: [T,H,W,C] with T typically smaller than H,W
        t, h, w, c = tensor.shape
        if t <= 64 and h >= c and w >= c:
            tensor = _thw_to_hwt(tensor)
    if expect_channels is not None and tensor.shape[-1] != expect_channels:
        raise ValueError(f"Expected {expect_channels} channels, got shape {tensor.shape}")
    return tensor


def cocoa_batch_to_galileo_input(
    batch_dict: dict[str, torch.Tensor | None],
    *,
    normalize: bool = True,
) -> MaskedOutput:
    """
    Map a cocoa training/inference batch dict to Galileo ``MaskedOutput``.

    Expected keys (all optional except at least one tensor present):

    - ``s2``: Sentinel-2 ``[T, H, W, len(S2_BANDS)]`` (see :data:`data.sentinel_composite.S2_OPTICAL_BANDS`)
    - ``s1``: Sentinel-1 VV/VH ``[T, H, W, 2]``
    - ``srtm``: elevation + slope ``[H, W, 2]``
    - ``era5``: precipitation + temperature ``[T, 2]``
    - ``terraclim``: aet + def + soil ``[T, 3]``
    - ``dynamic_world``: class probabilities ``[H, W, 9]``
    - ``world_cereal``: ag mask ``[H, W, 1]`` (mapped into WorldCrops channel 0)
    - ``location``: ``[lat, lon]`` or ``[..., 2]``
    - ``ndvi``, ``months``, ``viirs``, ``landscan``: optional Galileo modalities
    """
    s2 = _maybe_hwt(batch_dict.get("s2"), expect_channels=10)
    s1 = _maybe_hwt(batch_dict.get("s1"), expect_channels=2)
    ndvi = batch_dict.get("ndvi")
    if ndvi is not None and ndvi.ndim == 4:
        ndvi = _thw_to_hwt(ndvi.unsqueeze(-1)).squeeze(-1)

    srtm = batch_dict.get("srtm")
    dynamic_world = batch_dict.get("dynamic_world")
    world_cereal = batch_dict.get("world_cereal")
    era5 = batch_dict.get("era5")
    terraclim = batch_dict.get("terraclim")
    viirs = batch_dict.get("viirs")
    landscan = batch_dict.get("landscan")
    months = batch_dict.get("months")

    location = batch_dict.get("location")
    latlon: torch.Tensor | None = None
    if location is not None:
        latlon = location_to_galileo_xyz(location.float())

    wc: torch.Tensor | None = None
    if world_cereal is not None:
        if world_cereal.ndim == 3:
            wc = torch.zeros(
                (*world_cereal.shape[:2], 5),
                dtype=world_cereal.dtype,
                device=world_cereal.device,
            )
            wc[..., :1] = world_cereal[..., :1]
        else:
            wc = world_cereal

    cgi, _ = _resolve_galileo_utils()
    return cgi(
        s2=s2,
        s1=s1,
        ndvi=ndvi,
        srtm=srtm,
        dw=dynamic_world,
        wc=wc,
        era5=era5,
        tc=terraclim,
        viirs=viirs,
        landscan=landscan,
        latlon=latlon,
        months=months,
        normalize=normalize,
    )


def _temporal_mean_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Collapse ``[B,T,H,W,C]`` or ``[T,H,W,C]`` to ``[B,C,H,W]`` (mean over time)."""
    if tensor.dim() == 4:
        return tensor.mean(dim=0).permute(2, 0, 1).contiguous().unsqueeze(0)
    if tensor.dim() == 5:
        return tensor.mean(dim=1).permute(0, 3, 1, 2).contiguous()
    if tensor.dim() == 3:
        return tensor.permute(2, 0, 1).unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] <= 20:
        return tensor
    raise ValueError(f"Cannot collapse tensor shape {tuple(tensor.shape)} to BCHW")


def _pad_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
    if x.shape[1] >= target_c:
        return x[:, :target_c]
    pad = target_c - x.shape[1]
    zeros = torch.zeros(x.shape[0], pad, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=1)


def cocoa_batch_to_terramind_input(
    batch_dict: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    """
    Map a cocoa batch dict to TerraMind modality tensors ``[B, C, H, W]``.

    Keys: ``S2L2A`` (12-band), ``S1GRD`` (VV/VH), ``DEM`` (elevation + slope from ``srtm``).
    """
    from models.terramind_backbone import DEM_CHANNELS, S1GRD_CHANNELS, S2L2A_CHANNELS

    out: dict[str, torch.Tensor] = {}
    s2 = batch_dict.get("s2")
    if s2 is not None:
        s2_b = _temporal_mean_bchw(s2.float())
        out["S2L2A"] = _pad_channels(s2_b, S2L2A_CHANNELS)

    s1 = batch_dict.get("s1")
    if s1 is not None:
        s1_b = _temporal_mean_bchw(s1.float())
        out["S1GRD"] = _pad_channels(s1_b, S1GRD_CHANNELS)

    dem = batch_dict.get("srtm") or batch_dict.get("dem")
    if dem is not None:
        if dem.dim() == 3:
            dem = dem.unsqueeze(0).permute(0, 3, 1, 2)
        elif dem.dim() == 4 and dem.shape[-1] <= 4:
            dem = dem.permute(0, 3, 1, 2)
        out["DEM"] = _pad_channels(dem.float(), DEM_CHANNELS)

    if not out:
        raise ValueError("batch_dict must include at least one of s2, s1, srtm/dem for TerraMind")
    return out


__all__ = [
    "MaskedOutput",
    "cocoa_batch_to_galileo_input",
    "cocoa_batch_to_terramind_input",
    "construct_galileo_input",
    "location_to_galileo_xyz",
]


def __getattr__(name: str) -> Any:
    if name == "MaskedOutput":
        _, masked_output = _resolve_galileo_utils()
        return masked_output
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
