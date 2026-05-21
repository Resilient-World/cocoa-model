"""API configuration from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Re-export for settings validation
ExposureBackend = Literal["fdp", "galileo", "aef", "agrifm", "ensemble", "ensemble_v2"]

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from data.cocoa_exposure import DEFAULT_THRESHOLD
from models.cqr import DEFAULT_CQR_CALIBRATOR, DEFAULT_CQR_CHECKPOINT

_REPO_ROOT = Path(__file__).resolve().parents[2]

UQMethod = Literal["mcd", "cqr"]

ConformalMethod = Literal[
    "split_cqr",
    "aci",
    "conformal_pid",
    "eci",
    "eci_integral",
]


class APISettings(BaseSettings):
    """Settings for the FastAPI inference service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    yield_surrogate_version: Literal["v1", "v2"] = Field(
        default="v2",
        validation_alias="YIELD_SURROGATE_VERSION",
    )
    model_checkpoint_path: str | None = Field(
        default=None,
        validation_alias="MODEL_CHECKPOINT_PATH",
    )
    allow_v1_fallback: bool = Field(
        default=True,
        validation_alias="YIELD_SURROGATE_ALLOW_V1_FALLBACK",
        description="When v2 checkpoint is missing, load v1 weights via from_v1_checkpoint",
    )
    enable_cssvd_landscape: bool = Field(
        default=False,
        validation_alias="ENABLE_CSSVD_LANDSCAPE",
        description="Use Dumont et al. landscape CSSVD incidence model when checkpoint exists",
    )
    cssvd_landscape_checkpoint: Path = Field(
        default=_REPO_ROOT / "models" / "cssvd_landscape.joblib",
        validation_alias="CSSVD_LANDSCAPE_CHECKPOINT",
    )
    enable_teleconnection: bool = Field(
        default=True,
        validation_alias="ENABLE_TELECONNECTION",
    )
    teleconnection_parquet_path: Path = Field(
        default=_REPO_ROOT / "data" / "external" / "teleconnection_indices.parquet",
        validation_alias="TELECONNECTION_PARQUET_PATH",
    )
    teleconnection_checkpoint_path: Path = Field(
        default=_REPO_ROOT / "models" / "yield_surrogate_v2_teleconnection.pt",
        validation_alias="TELECONNECTION_CHECKPOINT_PATH",
    )
    scenario_yield_backend: Literal["v2_teleconnection", "casej"] = Field(
        default="v2_teleconnection",
        validation_alias="SCENARIO_YIELD_BACKEND",
    )
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

    conformal_method: ConformalMethod = Field(
        default="eci_integral",
        validation_alias="CONFORMAL_METHOD",
        description="Online conformal for /simulate-scenario (split_cqr = static CQR)",
    )
    online_conformal_state_path: Path = Field(
        default=_REPO_ROOT / "data" / "processed" / "online_conformal_state.json",
        validation_alias="ONLINE_CONFORMAL_STATE_PATH",
    )
    conformal_initial_state_path: Path = Field(
        default=_REPO_ROOT / "data" / "processed" / "conformal_initial_state.json",
        validation_alias="CONFORMAL_INITIAL_STATE_PATH",
    )
    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")
    conformal_alpha: float = Field(default=0.1, validation_alias="CONFORMAL_ALPHA")
    eci_eta: float = Field(default=2.5, validation_alias="ECI_ETA")
    eci_decay: float = Field(default=0.95, validation_alias="ECI_DECAY")
    eci_window: int = Field(default=100, validation_alias="ECI_WINDOW")
    aci_eta: float = Field(default=0.005, validation_alias="ACI_ETA")
    pid_eta: float = Field(default=0.01, validation_alias="PID_ETA")

    @model_validator(mode="after")
    def _default_model_checkpoint(self) -> APISettings:
        if self.model_checkpoint_path is None:
            if self.yield_surrogate_version == "v2":
                object.__setattr__(self, "model_checkpoint_path", "models/yield_surrogate_v2.pt")
            else:
                object.__setattr__(self, "model_checkpoint_path", "models/yield_surrogate_v1.pt")
        return self

    def resolved_uq_method(self) -> UQMethod:
        """Use CQR when calibrator exists; otherwise fall back to MCD."""
        if self.uq_method == "cqr" and self.cqr_calibrator_path.is_file():
            return "cqr"
        if self.uq_method == "cqr" and not self.cqr_calibrator_path.is_file():
            return "mcd"
        return self.uq_method
