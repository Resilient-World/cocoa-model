"""Champion / challenger MLflow Model Registry aliases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

_REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_EVIDENCE_DIR = _REPO_ROOT / "release_evidence"
ROLLBACK_FILENAME = "rollback_target.json"


@dataclass(frozen=True)
class AliasSnapshot:
    version: str | None
    run_id: str | None


def _client() -> MlflowClient:
    return MlflowClient()


def _get_alias_version(client: MlflowClient, model_name: str, alias: str) -> AliasSnapshot:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return AliasSnapshot(version=str(mv.version), run_id=mv.run_id)
    except MlflowException:
        return AliasSnapshot(version=None, run_id=None)


def register_model(
    model_name: str,
    run_id: str,
    *,
    alias: str = "challenger",
    artifact_path: str = "model",
) -> str:
    """
    Register an MLflow run as a new model version and set ``alias`` (default challenger).
    """
    client = _client()
    try:
        client.create_registered_model(model_name)
    except MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
    version = str(mv.version)
    client.set_registered_model_alias(model_name, alias, version)
    return version


def get_champion_version(model_name: str) -> AliasSnapshot:
    return _get_alias_version(_client(), model_name, "champion")


def get_champion(model_name: str) -> Any:
    """Load champion PyFunc model (``models:/name@champion``)."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")


def promote_challenger(
    model_name: str,
    *,
    gate_result: dict[str, Any] | None = None,
    env: str = "staging",
) -> dict[str, Any]:
    """
    Atomically move challenger → champion; previous champion recorded for rollback.
    """
    from registry.release_evidence import write_release_bundle

    client = _client()
    prev_champion = _get_alias_version(client, model_name, "champion")
    challenger = _get_alias_version(client, model_name, "challenger")
    if challenger.version is None:
        raise MlflowException(f"No challenger alias on registered model {model_name!r}")

    client.set_registered_model_alias(model_name, "champion", challenger.version)

    rollback_payload = {
        "model_name": model_name,
        "previous_champion_version": prev_champion.version,
        "previous_champion_run_id": prev_champion.run_id,
        "new_champion_version": challenger.version,
        "new_champion_run_id": challenger.run_id,
    }
    RELEASE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASE_EVIDENCE_DIR / ROLLBACK_FILENAME).write_text(
        json.dumps(rollback_payload, indent=2),
        encoding="utf-8",
    )

    bundle = write_release_bundle(
        model_name=model_name,
        run_id=challenger.run_id or "",
        env=env,
        gate_result=gate_result or {},
        rollback=rollback_payload,
    )
    return bundle


def rollback(model_name: str) -> str:
    """Restore champion alias from ``release_evidence/rollback_target.json``."""
    path = RELEASE_EVIDENCE_DIR / ROLLBACK_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing rollback metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("previous_champion_version")
    if not version:
        raise ValueError("rollback_target.json has no previous_champion_version")
    client = _client()
    client.set_registered_model_alias(model_name, "champion", str(version))
    return str(version)
