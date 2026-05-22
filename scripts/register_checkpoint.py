#!/usr/bin/env python3
"""Register a local checkpoint as MLflow challenger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from registry.mlflow_pyfunc import YieldSurrogatePyfunc
from registry.mlflow_registry import register_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register checkpoint as MLflow challenger")
    parser.add_argument("--model-name", default="yield_surrogate_v2")
    parser.add_argument(
        "--checkpoint", type=Path, default=_REPO_ROOT / "models" / "yield_surrogate_v2.pt"
    )
    parser.add_argument("--experiment", default="resilient-cocoa-registry")
    args = parser.parse_args(argv)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YieldSurrogatePyfunc(),
            artifacts={"checkpoint": str(args.checkpoint)},
        )
        version = register_model(args.model_name, run.info.run_id, alias="challenger")
        print(
            f"Registered {args.model_name} version {version} as challenger (run {run.info.run_id})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
