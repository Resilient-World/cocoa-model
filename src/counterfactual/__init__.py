"""
Counterfactual climate (ATTRICI subprocess boundary, Zarr providers).

ATTRICI is GPLv3 and is never imported from this package — see
:mod:`counterfactual.attrici_runner`.
"""

from counterfactual.attrici_runner import (
    ATTRICIRunner,
    CounterfactualClimateProvider,
    SUPPORTED_VARIABLES,
    ZarrCounterfactualProvider,
    load_counterfactual,
)

__all__ = [
    "ATTRICIRunner",
    "CounterfactualClimateProvider",
    "SUPPORTED_VARIABLES",
    "ZarrCounterfactualProvider",
    "load_counterfactual",
]
