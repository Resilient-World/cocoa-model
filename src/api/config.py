"""API configuration from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Re-export for settings validation
ExposureBackend = Literal["fdp", "galileo", "aef", "agrifm", "ensemble", "ensemble_v2"]

from pydantic import Field, field_validator, model_validator
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
    mlflow_registry_enabled: bool = Field(
        default=False,
        validation_alias="MLFLOW_REGISTRY_ENABLED",
        description="Load yield surrogate from models:/<name>@champion when set",
    )
    mlflow_registry_model_name: str = Field(
        default="yield_surrogate_v2",
        validation_alias="MLFLOW_REGISTRY_MODEL_NAME",
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
    mediation_n_bootstrap: int = Field(
        default=200,
        ge=50,
        le=500,
        validation_alias="MEDIATION_N_BOOTSTRAP",
        description="Bootstrap reps for intervention mediation (API latency cap)",
    )

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
    corrdiff_processed_dir: Path = Field(
        default=_REPO_ROOT / "data" / "processed",
        validation_alias="CORRDIFF_PROCESSED_DIR",
    )
    corrdiff_allow_inline: bool = Field(
        default=False,
        validation_alias="CORRDIFF_ALLOW_INLINE",
        description="Dev-only: run CorrDiff on cache miss (requires GPU + [corrdiff] extra)",
    )
    corrdiff_source_id: str = Field(default="CanESM5", validation_alias="CORRDIFF_SOURCE_ID")
    corrdiff_variant_label: str = Field(
        default="r1i1p2f1", validation_alias="CORRDIFF_VARIANT_LABEL"
    )
    corrdiff_number_of_samples: int = Field(
        default=8, validation_alias="CORRDIFF_NUMBER_OF_SAMPLES"
    )
    corrdiff_solver: Literal["euler", "heun"] = Field(
        default="euler", validation_alias="CORRDIFF_SOLVER"
    )
    corrdiff_sampler_type: Literal["deterministic", "stochastic"] = Field(
        default="stochastic",
        validation_alias="CORRDIFF_SAMPLER_TYPE",
    )
    static_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "site_static.zarr"
    feature_cache_dir: Path = _REPO_ROOT / "data" / "cache" / "api_features"
    feature_store_root: Path | None = None
    climate_reference_year: int = 2023
    earthengine_project: str | None = None

    whisp_base_url: str = "https://whisp.openforis.org"
    whisp_api_key: str | None = None

    cocoa_exposure_year: int = 2023
    cocoa_exposure_backend: Literal[
        "fdp",
        "galileo",
        "aef",
        "agrifm",
        "terramind",
        "terramind_tim",
        "ensemble",
        "ensemble_v2",
        "ensemble_v3",
    ] = Field(default="ensemble_v2", validation_alias="COCOA_EXPOSURE_BACKEND")
    ensemble_backend: Literal["v1", "v2", "v3"] = Field(
        default="v2",
        validation_alias="ENSEMBLE_BACKEND",
        description="v2 → ensemble_v2 weights; v3 → ensemble_v3 (AEF+Galileo+AgriFM+TerraMind+FDP)",
    )
    ensemble_weights_path: Path = _REPO_ROOT / "config" / "ensemble_weights.yaml"
    ensemble_v3_weights_path: Path = _REPO_ROOT / "config" / "ensemble_weights_v3.yaml"
    agrifm_checkpoint_path: Path = _REPO_ROOT / "models" / "agrifm_cocoa_seg.pt"
    terramind_checkpoint_path: Path = _REPO_ROOT / "models" / "terramind_cocoa_seg.pt"
    terramind_tim_checkpoint_path: Path = _REPO_ROOT / "models" / "terramind_tim_cocoa_seg.pt"
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

    drift_enabled: bool = Field(default=True, validation_alias="DRIFT_ENABLED")
    drift_state_path: Path = Field(
        default=_REPO_ROOT / "data" / "processed" / "drift_monitoring_state.json",
        validation_alias="DRIFT_STATE_PATH",
    )
    drift_alpha_fpr: float = Field(default=0.01, validation_alias="DRIFT_ALPHA_FPR")
    drift_inflation_factor: float = Field(
        default=1.5,
        validation_alias="DRIFT_INFLATION_FACTOR",
    )
    drift_score_cap: float = Field(default=8.0, validation_alias="DRIFT_SCORE_CAP")
    drift_cusum_h: float = Field(default=5.0, validation_alias="DRIFT_CUSUM_H")
    drift_cusum_k: float = Field(default=0.0, validation_alias="DRIFT_CUSUM_K")

    farm_panel_parquet_path: Path = Field(
        default=_REPO_ROOT / "data" / "raw" / "farm_panel.parquet",
        validation_alias="FARM_PANEL_PARQUET_PATH",
    )
    dvds_lambda_grid: list[float] = Field(
        default_factory=lambda: [1.1, 1.25, 1.5, 2.0],
        validation_alias="DVDS_LAMBDA_GRID",
    )

    otel_enabled: bool = Field(default=False, validation_alias="OTEL_ENABLED")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_service_name: str = Field(
        default="cocoa-model-api",
        validation_alias="OTEL_SERVICE_NAME",
    )
    otel_service_version: str = Field(
        default="0.3.0",
        validation_alias="OTEL_SERVICE_VERSION",
    )
    otel_deployment_environment: str = Field(
        default="local",
        validation_alias="OTEL_DEPLOYMENT_ENVIRONMENT",
    )
    prometheus_enabled: bool = Field(default=False, validation_alias="PROMETHEUS_ENABLED")
    metrics_auth_token: str | None = Field(default=None, validation_alias="METRICS_AUTH_TOKEN")
    prometheus_metrics_path: str = Field(
        default="/metrics",
        validation_alias="PROMETHEUS_METRICS_PATH",
    )

    interpret_enabled: bool = Field(default=False, validation_alias="INTERPRET_ENABLED")
    interpret_auth_token: str | None = Field(default=None, validation_alias="INTERPRET_AUTH_TOKEN")

    neuralgcm_enabled: bool = Field(default=False, validation_alias="NEURALGCM_ENABLED")
    ace2_era5_enabled: bool = Field(default=False, validation_alias="ACE2_ERA5_ENABLED")

    process_bma_enabled: bool = Field(default=False, validation_alias="PROCESS_BMA_ENABLED")
    ensemble_process_method: str = Field(
        default="mean",
        validation_alias="ENSEMBLE_PROCESS_METHOD",
        description="mean | bma | best for CASEJ/CASE2/ALMANAC on simulate-scenario",
    )
    process_bma_weights_path: Path = Field(
        default=Path("config/process_bma_weights.json"),
        validation_alias="PROCESS_BMA_WEIGHTS_PATH",
    )

    @field_validator("dvds_lambda_grid", mode="before")
    @classmethod
    def _parse_dvds_lambda_grid(cls, value: object) -> object:
        if isinstance(value, str):
            return [float(x.strip()) for x in value.split(",") if x.strip()]
        return value

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
