"""CLI: ``python -m registry.promote --model NAME``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry.mlflow_registry import promote_challenger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote MLflow challenger to champion")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate-result", type=Path, default=Path("release_evidence/gate_result.json"))
    parser.add_argument("--env", default="staging")
    args = parser.parse_args(argv)
    gate: dict = {}
    if args.gate_result.is_file():
        gate = json.loads(args.gate_result.read_text(encoding="utf-8"))
    bundle = promote_challenger(args.model, gate_result=gate, env=args.env)
    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
