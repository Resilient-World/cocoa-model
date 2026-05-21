"""Biotic yield-loss models for West African cocoa."""

from hazards.black_pod import BlackPodRiskModel, ShadeSpecies
from hazards.composite import apply_biotic_losses
from hazards.cssvd import CSSVDRiskModel
from hazards.cssvd_landscape import IncidencePrediction, LandscapeCSSVDModel
from hazards.mirids import MiridPressureModel

__all__ = [
    "BlackPodRiskModel",
    "CSSVDRiskModel",
    "IncidencePrediction",
    "LandscapeCSSVDModel",
    "MiridPressureModel",
    "ShadeSpecies",
    "apply_biotic_losses",
]
