"""Tests for MLflow champion/challenger registry (local file store)."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from registry.mlflow_pyfunc import YieldSurrogatePyfunc
from registry.mlflow_registry import (
    get_champion_version,
    promote_challenger,
    register_model,
    rollback,
)


@pytest.fixture()
def mlflow_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tmp_path


def test_register_and_promote(mlflow_tmp: Path, tmp_path: Path) -> None:
    ckpt = tmp_path / "yield.pt"
    import torch

    from models.yield_surrogate_v2 import YieldSurrogateV2

    model = YieldSurrogateV2()
    torch.save({"state_dict": model.state_dict(), "version": "v2"}, ckpt)

    mlflow.set_experiment("test_registry")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YieldSurrogatePyfunc(),
            artifacts={"checkpoint": str(ckpt)},
        )
        register_model("test_yield", run.info.run_id, alias="challenger")

    bundle = promote_challenger("test_yield", gate_result={"checks": {"crps": True}}, env="test")
    assert "release_dir" in bundle
    champ = get_champion_version("test_yield")
    assert champ.version is not None

    rollback("test_yield")
    rb = json.loads((Path("release_evidence") / "rollback_target.json").read_text())
    assert rb["model_name"] == "test_yield"
