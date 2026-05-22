"""Schema defaults for opt-in backends (source inspection, no GEE import chain)."""

from __future__ import annotations

from pathlib import Path

_SCHEMAS = Path(__file__).resolve().parents[1] / "src" / "api" / "schemas.py"


def test_downscaling_and_process_defaults_in_schema_source() -> None:
    text = _SCHEMAS.read_text(encoding="utf-8")
    assert 'DownscalingMethod = Literal["linear_delta", "corrdiff", "neuralgcm", "ace2_era5"]' in text
    assert 'ensemble_process_method: ProcessEnsembleMethod = Field(\n        default="mean"' in text
    assert 'downscaling_method: DownscalingMethod = Field(\n        default="linear_delta"' in text
