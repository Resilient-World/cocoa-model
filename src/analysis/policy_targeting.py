"""
Policy targeting utilities for cooperative-level rollout planning.

These functions are matplotlib-free: they return DataFrames that can be plotted
by the caller (frontend/notebooks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.heterogeneity import CATEResult


def rank_farms_by_uplift(
    cate_result: CATEResult,
    *,
    intervention_cost_usd_per_farm: float,
    cocoa_price_usd: float,
    farm_areas_ha: pd.Series,
) -> pd.DataFrame:
    """
    Rank farms by expected net uplift USD.

    Net uplift (USD) = max(tau_hat, 0) * area_ha * cocoa_price_usd - intervention_cost
    Ties are broken by lower standard error (more certain uplift first).
    """
    tau = cate_result.tau_hat
    se = cate_result.se.reindex(tau.index)
    area = farm_areas_ha.reindex(tau.index).astype(float)

    avoided_tonnes = np.maximum(tau.to_numpy(dtype=float), 0.0) * area.to_numpy(dtype=float)
    gross_usd = avoided_tonnes * float(cocoa_price_usd)
    net_usd = gross_usd - float(intervention_cost_usd_per_farm)

    out = pd.DataFrame(
        {
            "tau_hat_tonnes_per_ha": tau.astype(float),
            "se": se.astype(float),
            "area_ha": area,
            "avoided_loss_tonnes": avoided_tonnes,
            "gross_uplift_usd": gross_usd,
            "net_uplift_usd": net_usd,
        },
        index=tau.index,
    )
    return out.sort_values(by=["net_uplift_usd", "se"], ascending=[False, True])


def policy_value_curve(
    ranked: pd.DataFrame,
    *,
    uplift_col: str = "avoided_loss_tonnes",
) -> pd.DataFrame:
    """
    Cumulative value curve for targeting the top-K farms.

    Parameters
    ----------
    ranked:
        Output from :func:`rank_farms_by_uplift` (already sorted).
    uplift_col:
        Column to cumulate, e.g. ``avoided_loss_tonnes`` or ``net_uplift_usd``.
    """
    if uplift_col not in ranked.columns:
        raise ValueError(f"Missing uplift column '{uplift_col}'")
    vals = ranked[uplift_col].to_numpy(dtype=float)
    cum = np.cumsum(np.maximum(vals, 0.0))
    return pd.DataFrame(
        {
            "k": np.arange(1, len(ranked) + 1),
            "cumulative_value": cum,
        }
    )


__all__ = ["rank_farms_by_uplift", "policy_value_curve"]

