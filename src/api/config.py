"""API configuration from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from data.cocoa_exposure import DEFAULT_THRESHOLD

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
    mc_num_samples: int = 50
    yield_blend_weight: float = 0.0

    # Feature resolution (replaces geo_mock on the simulation hot path)
    era5_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
    cmip6_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "cmip6_ensemble.zarr"
    static_zarr_path: Path = _REPO_ROOT / "data" / "processed" / "site_static.zarr"
    feature_cache_dir: Path = _REPO_ROOT / "data" / "cache" / "api_features"
    feature_store_root: Path | None = None
    climate_reference_year: int = 2023
    earthengine_project: str | None = None

    cocoa_exposure_year: int = 2023
    cocoa_exposure_threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.5,
        le=1.0,
        description="FDP 2025a probability threshold for binary cocoa mask",
    )

    use_galileo_embedding: bool = False
    galileo_embedding_dim: int = 128

    conformal_json_path: Path = _REPO_ROOT / "models" / "conformal.json"
