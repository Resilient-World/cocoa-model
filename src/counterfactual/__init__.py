"""
Counterfactual climate (ATTRICI subprocess boundary, Zarr providers).

ATTRICI is GPLv3 and is never imported from this package — see
:mod:`counterfactual.attrici_runner`.
"""

from counterfactual.attrici_runner import (
    SUPPORTED_VARIABLES,
    ATTRICIRunner,
    CounterfactualClimateProvider,
    ZarrCounterfactualProvider,
    load_counterfactual,
)
from counterfactual.corrdiff_downscaler import (
    CorrDiffCMIP6Downscaler,
    corrdiff_cache_path,
)

__all__ = [
    "SUPPORTED_VARIABLES",
    "ATTRICIRunner",
    "CorrDiffCMIP6Downscaler",
    "CounterfactualClimateProvider",
    "ZarrCounterfactualProvider",
    "corrdiff_cache_path",
    "load_counterfactual",
]
