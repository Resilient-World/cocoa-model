"""
Composite biotic yield-loss layer (black pod + CSSVD + mirids).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hazards.black_pod import BlackPodRiskModel, ShadeSpecies
from hazards.cssvd import CSSVDRiskModel
from hazards.mirids import MiridPressureModel

MIN_SURVIVING_FRACTION = 0.30  # cap combined loss at 70%

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSSVD_LANDSCAPE_CKPT = _REPO_ROOT / "models" / "cssvd_landscape.joblib"


def _resolve_shade_species(static_features: dict[str, Any]) -> ShadeSpecies | None:
    raw = static_features.get("shade_species")
    if raw is None:
        return ShadeSpecies.UNSHADED
    if isinstance(raw, ShadeSpecies):
        return raw
    return ShadeSpecies(str(raw))


def _landscape_checkpoint_path() -> Path | None:
    raw = os.environ.get("CSSVD_LANDSCAPE_CHECKPOINT", "").strip()
    if raw:
        return Path(raw)
    if os.environ.get("ENABLE_CSSVD_LANDSCAPE", "").lower() in ("1", "true", "yes"):
        if _DEFAULT_CSSVD_LANDSCAPE_CKPT.is_file():
            return _DEFAULT_CSSVD_LANDSCAPE_CKPT
    return None


def _cssvd_loss(
    static_features: dict[str, Any],
    cssvd_model: CSSVDRiskModel,
) -> tuple[float, dict[str, Any] | None]:
    """
    CSSVD yield-loss fraction and optional landscape incidence metadata.
    """
    ckpt = static_features.get("cssvd_landscape_checkpoint")
    if ckpt is None:
        ckpt = _landscape_checkpoint_path()
    use_landscape = static_features.get("use_cssvd_landscape")
    if use_landscape is None and ckpt is not None:
        use_landscape = True

    lat = static_features.get("lat")
    lon = static_features.get("lon")
    year = int(static_features.get("year", 2023))
    tolerance = float(static_features.get("cssvd_tolerance", 1.0))
    conservative = bool(static_features.get("cssvd_conservative", False))

    precomputed = static_features.get("cssvd_landscape_features")
    if use_landscape and precomputed is not None:
        path = Path(str(ckpt)) if ckpt else _DEFAULT_CSSVD_LANDSCAPE_CKPT
        if path.is_file():
            from hazards.cssvd_landscape import LandscapeCSSVDModel

            lm = cssvd_model.landscape_model or LandscapeCSSVDModel.from_checkpoint(path)
            inc = lm.predict_from_features(dict(precomputed))
            prob = inc.pi_high if conservative else inc.point
            loss = cssvd_model.annual_yield_loss_fraction(100.0 * prob, tolerance=tolerance)
            return loss, {
                "cssvd_incidence_prob_12mo": inc.point,
                "cssvd_incidence_pi_low": inc.pi_low,
                "cssvd_incidence_pi_high": inc.pi_high,
            }

    if use_landscape and lat is not None and lon is not None:
        path = Path(str(ckpt)) if ckpt else _DEFAULT_CSSVD_LANDSCAPE_CKPT
        if path.is_file():
            model = cssvd_model.landscape_model
            if model is None:
                model = CSSVDRiskModel.with_landscape_checkpoint(path).landscape_model
            loss, inc = cssvd_model.annual_yield_loss_from_landscape(
                float(lat),
                float(lon),
                year,
                tolerance=tolerance,
                conservative=conservative,
                landscape_model=model,
            )
            meta = {
                "cssvd_incidence_prob_12mo": inc.point,
                "cssvd_incidence_pi_low": inc.pi_low,
                "cssvd_incidence_pi_high": inc.pi_high,
            }
            return loss, meta

    prevalence = float(static_features.get("cssvd_prevalence_pct", 15.0))
    loss = cssvd_model.annual_yield_loss_fraction(prevalence, tolerance=tolerance)
    return loss, None


def apply_biotic_losses(
    climate_yield_t_ha: float,
    ds: xr.Dataset,
    static_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply independent multiplicative biotic survival factors to climate-only yield.

    Parameters
    ----------
    climate_yield_t_ha:
        Yield from the climate surrogate (tonnes/ha) before biotic adjustment.
    ds:
        Daily climate ``xr.Dataset`` with ``tmean``, ``rh_mean``, ``precip``.
    static_features:
        Farm static covariates, e.g. ``cssvd_prevalence_pct``, ``cssvd_tolerance``,
        ``shade_species``, optional ``lat``/``lon``/``year`` for landscape CSSVD.

    Returns
    -------
    dict with ``final_yield``, ``surviving_fraction``, and ``loss_attribution`` fractions.
    """
    shade = _resolve_shade_species(static_features)

    bp_model = BlackPodRiskModel()
    bp_loss = float(bp_model.seasonal_yield_loss_fraction(ds, shade_species=shade).values)

    cssvd_model = CSSVDRiskModel()
    cssvd_loss, cssvd_meta = _cssvd_loss(static_features, cssvd_model)

    mirid_model = MiridPressureModel()
    mirid_loss = mirid_model.annual_yield_loss_fraction(ds, shade_species=shade)

    surviving_fraction = (1.0 - bp_loss) * (1.0 - cssvd_loss) * (1.0 - mirid_loss)
    surviving_fraction = max(float(surviving_fraction), MIN_SURVIVING_FRACTION)

    final_yield = float(climate_yield_t_ha) * surviving_fraction
    total_loss_fraction = 1.0 - surviving_fraction

    result: dict[str, Any] = {
        "final_yield": final_yield,
        "climate_yield_t_ha": float(climate_yield_t_ha),
        "surviving_fraction": surviving_fraction,
        "total_loss_fraction": total_loss_fraction,
        "loss_attribution": {
            "black_pod": bp_loss,
            "cssvd": cssvd_loss,
            "mirids": mirid_loss,
        },
    }
    if cssvd_meta:
        result["cssvd_landscape"] = cssvd_meta
    return result


def estimate_surviving_biotic_fraction(
    ds: xr.Dataset,
    static_features: dict[str, Any] | None = None,
) -> float:
    """Multiplicative biotic survival used to back out pre-biotic yield targets."""
    features = static_features or {}
    return float(apply_biotic_losses(1.0, ds, features)["surviving_fraction"])


__all__ = [
    "apply_biotic_losses",
    "estimate_surviving_biotic_fraction",
    "MIN_SURVIVING_FRACTION",
]
