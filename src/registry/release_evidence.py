"""Release evidence bundle on model promotion."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_TEMPLATE = _REPO_ROOT / "docs" / "MODEL_CARD.md"
RELEASE_EVIDENCE_DIR = _REPO_ROOT / "release_evidence"
REPORTS_RELEASES = _REPO_ROOT / "reports" / "releases"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _config_hash() -> str:
    params = _REPO_ROOT / "params.yaml"
    if not params.is_file():
        return ""
    return hashlib.sha256(params.read_bytes()).hexdigest()[:16]


def _next_version_dir(model_name: str, env: str) -> Path:
    base = REPORTS_RELEASES / model_name / env
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")],
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0,
    )
    n = 1
    if existing:
        last = existing[-1].name
        if last[1:].isdigit():
            n = int(last[1:]) + 1
    return base / f"v{n}"


def _render_model_card(metrics: dict[str, Any]) -> str:
    template = (
        MODEL_CARD_TEMPLATE.read_text(encoding="utf-8")
        if MODEL_CARD_TEMPLATE.is_file()
        else "# Model card\n"
    )
    block = json.dumps(metrics, indent=2)
    return f"{template}\n\n---\n\n## Promotion metrics\n\n```json\n{block}\n```\n"


def write_release_bundle(
    *,
    model_name: str,
    run_id: str,
    env: str,
    gate_result: dict[str, Any],
    rollback: dict[str, Any],
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset_fingerprint": _file_hash(
            _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
        ),
        "config_hash": _config_hash(),
        "git_sha": _git_sha(),
        "env_hash": _git_sha(),
        "model_name": model_name,
        "run_id": run_id,
    }
    decision = {
        "timestamp": ts,
        "run_id": run_id,
        "model_name": model_name,
        "metrics": gate_result.get("metrics", {}),
        "gate_status": gate_result.get("checks", gate_result),
    }

    RELEASE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "promotion_decision.json": decision,
        "release_manifest.json": manifest,
        "rollback_target.json": rollback,
    }
    for name, payload in files.items():
        (RELEASE_EVIDENCE_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    card = _render_model_card(decision.get("metrics", {}))
    (RELEASE_EVIDENCE_DIR / "model_card.md").write_text(card, encoding="utf-8")

    version_dir = _next_version_dir(model_name, env)
    version_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        if name.endswith(".json"):
            (version_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (version_dir / "model_card.md").write_text(card, encoding="utf-8")

    return {"release_dir": str(version_dir), "timestamp": ts, "manifest": manifest}
