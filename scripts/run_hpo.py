#!/usr/bin/env python3
"""Run Optuna hyperparameter optimization with MLflow logging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna HPO entrypoint")
    parser.add_argument(
        "--model",
        choices=("yield", "galileo", "agrifm"),
        required=True,
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args(argv)

    if args.model == "yield":
        from training.optuna_yield import run_study

        run_study(n_trials=args.n_trials, study_name=args.study_name or "yield_hpo")
    elif args.model == "galileo":
        from training.optuna_galileo import run_study

        run_study(n_trials=args.n_trials, study_name=args.study_name or "galileo_hpo")
    else:
        from training.optuna_agrifm import run_study

        run_study(n_trials=args.n_trials, study_name=args.study_name or "agrifm_hpo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
