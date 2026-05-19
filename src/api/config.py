"""API configuration from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Settings for the FastAPI inference service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_checkpoint_path: str | None = None
    mc_num_samples: int = 50
    yield_blend_weight: float = 0.3
