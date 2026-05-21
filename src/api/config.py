"""API configuration from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Re-export for settings validation
ExposureBackend = Literal["fdp", "galileo", "aef", "agrifm", "ensemble", "ensemble_v2"]

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from data.cocoa_exposure import DEFAULT_THRESHOLD
from models.cqr import DEFAULT_CQR_CALIBRATOR, DEFAULT_CQR_CHECKPOINT

_REPO_ROOT = Path(__file__).resolve().parents[2]

UQMethod = Literal["mcd", "cqr"]


class APISettings(BaseSettings):
    """Settings for the FastAPI inference service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_checkpoint_path: str = "models/yield_surrogate_v1.pt"
    casej_checkpoint_path: str = "models/casej_surrogate.pt"
    mc_num_samples: int = 50
    yield_blend_weight: float = 0.0

    # Feature resolution (geo_mock only when use_real_features=false)
    use_real_features: bool = Field(
        default=True,
        validation_alias="USE_REAL_FEATURES",
        description="Use ERA5/static Zarr + feature cache; false → api.geo_mock (tests)",
    )
    era5_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
    features_cache_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "features_cache.zarr"
    era5_counterfactual_zarr_path: Path = (
        _REPO_ROOT / "data" / "processed" / "era5_counterfactual" / "civ_ghana_1980_2024.zarr"
    )
    attrici_gmt_file: Path = _REPO_ROOT / "data" / "raw" / "gmt" / "ssa_gmt.nc"
    attrici_cache_dir: Path = _REPO_ROOT / "data" / "cache" / "attrici_counterfactual"
    cmip6_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "cmip6_ensemble.zarr"
    static_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "site_static.zarr"
    feature_cache_dir: Path = _REPO_ROOT / "data" / "cache" / "api_features"
    feature_store_root: Path | None = None
    climate_reference_year: int = 2023
    earthengine_project: str | None = None

    whisp_base_url: str = "https://whisp.openforis.org"
    whisp_api_key: str | None = None

    cocoa_exposure_year: int = 2023
    cocoa_exposure_backend: Literal[
        "fdp", "galileo", "aef", "agrifm", "ensemble", "ensemble_v2"
    ] = Field(default="ensemble_v2", validation_alias="COCOA_EXPOSURE_BACKEND")
    ensemble_backend: Literal["v1", "v2"] = Field(
        default="v2",
        validation_alias="ENSEMBLE_BACKEND",
        description="Use ensemble_v2 weights when backend is ensemble/ensemble_v2",
    )
    ensemble_weights_path: Path = _REPO_ROOT / "config" / "ensemble_weights.yaml"
    agrifm_checkpoint_path: Path = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"
    cocoa_exposure_threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.5,
        le=1.0,
        description="FDP 2025a probability threshold for binary cocoa mask",
    )

    use_galileo_embedding: bool = False
    galileo_embedding_dim: int = 128

    conformal_json_path: Path = _REPO_ROOT / "models" / "conformal.json"

    uq_method: UQMethod = Field(
        default="cqr",
        description="Primary UQ: cqr (conformalized quantile regression) or mcd (MC dropout)",
    )
    cqr_checkpoint_path: Path = DEFAULT_CQR_CHECKPOINT
    cqr_calibrator_path: Path = DEFAULT_CQR_CALIBRATOR

    def resolved_uq_method(self) -> UQMethod:
        """Use CQR when calibrator exists; otherwise fall back to MCD."""
        if self.uq_method == "cqr" and self.cqr_calibrator_path.is_file():
            return "cqr"
        if self.uq_method == "cqr" and not self.cqr_calibrator_path.is_file():
            return "mcd"
        return self.uq_method
