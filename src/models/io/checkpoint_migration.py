"""
Checkpoint migration helpers for evolving static-feature layouts.

v1 → v2: 10 site static features → 13 (tree age / cohort / planting density).
Zero-pads new static MLP input columns and the extra stress-summary fusion column so
loaded v1 weights reproduce prior forward behaviour (f_age channel weight = 0).
"""

from __future__ import annotations

import structlog

from typing import Any

import torch
from torch import Tensor

log = structlog.get_logger(__name__)

V1_SITE_STATIC = 10
V2_SITE_STATIC = 13
V1_STRESS_SUMMARY_DIM = 4
V2_STRESS_SUMMARY_DIM = 5


def _pad_linear_input_weight(
    weight: Tensor,
    *,
    old_in: int,
    new_in: int,
    site_pad_start: int = V1_SITE_STATIC,
    site_pad_end: int = V2_SITE_STATIC,
) -> Tensor:
    """
    Expand ``[out, old_in]`` → ``[out, new_in]``.

    Inserts zero columns for new site static indices ``[10:13)``; preserves any
    Galileo tail after the site block unchanged.
    """
    if weight.ndim != 2 or weight.shape[1] != old_in:
        return weight
    site_tail = old_in - site_pad_start
    galileo_tail = max(0, site_tail - V1_SITE_STATIC)
    new_w = weight.new_zeros(weight.shape[0], new_in)
    new_w[:, :site_pad_start] = weight[:, :site_pad_start]
    if galileo_tail > 0:
        new_w[:, site_pad_end : site_pad_end + galileo_tail] = weight[:, site_pad_start:]
    return new_w


def _pad_residual_fusion_weight(
    weight: Tensor,
    *,
    gru_out_dim: int,
    static_hidden: int,
) -> Tensor:
    """Insert a zero column for the new stress-summary (f_age) dimension."""
    old_in = weight.shape[1]
    stress_start = gru_out_dim + static_hidden
    old_stress = V1_STRESS_SUMMARY_DIM
    new_stress = V2_STRESS_SUMMARY_DIM
    if old_in != stress_start + old_stress:
        return weight
    new_in = stress_start + new_stress
    new_w = weight.new_zeros(weight.shape[0], new_in)
    new_w[:, : stress_start + old_stress] = weight
    return new_w


def is_v1_static_checkpoint(state_dict: dict[str, Any]) -> bool:
    """True when ``static_mlp`` expects the legacy 10-dimensional site static block."""
    w = state_dict.get("static_mlp.0.weight")
    if not isinstance(w, Tensor):
        return False
    in_dim = int(w.shape[1])
    if in_dim == V1_SITE_STATIC:
        return True
    if in_dim > V1_SITE_STATIC and in_dim < V2_SITE_STATIC:
        return True
    return False


def migrate_v1_static_to_v2(
    state_dict: dict[str, Any],
    *,
    gru_out_dim: int = 192,
    static_hidden: int = 64,
) -> dict[str, Any]:
    """
    Upgrade a v1 checkpoint state dict to the 13-feature static layout.

    Parameters
    ----------
    state_dict:
        Raw or nested ``state_dict`` from ``torch.load``.
    gru_out_dim, static_hidden:
        Must match the trained :class:`~models.surrogate.yield_surrogate.YieldSurrogateModel`
        architecture (default bidirectional GRU hidden 96 → 192).
    """
    out = dict(state_dict)
    w_key = "static_mlp.0.weight"
    w = out.get(w_key)
    if isinstance(w, Tensor) and is_v1_static_checkpoint(out):
        old_in = int(w.shape[1])
        galileo_tail = max(0, old_in - V1_SITE_STATIC)
        new_in = V2_SITE_STATIC + galileo_tail
        out[w_key] = _pad_linear_input_weight(
            w,
            old_in=old_in,
            new_in=new_in,
            site_pad_start=V1_SITE_STATIC,
            site_pad_end=V2_SITE_STATIC,
        )
        log.info(
            "Migrated static_mlp input %d → %d (zero-padded tree-age cohort features)",
            old_in,
            new_in,
        )

    rh_w = out.get("residual_head.0.weight")
    if isinstance(rh_w, Tensor):
        expected_old = gru_out_dim + static_hidden + V1_STRESS_SUMMARY_DIM
        if int(rh_w.shape[1]) == expected_old:
            out["residual_head.0.weight"] = _pad_residual_fusion_weight(
                rh_w,
                gru_out_dim=gru_out_dim,
                static_hidden=static_hidden,
            )
            log.info("Migrated residual_head fusion for f_age stress channel")

    return out


__all__ = [
    "V1_SITE_STATIC",
    "V2_SITE_STATIC",
    "is_v1_static_checkpoint",
    "migrate_v1_static_to_v2",
]
