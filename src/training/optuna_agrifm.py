"""Optuna HPO for AgriFM cocoa segmentation (smoke / tile F1 proxy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import optuna
import structlog

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

log = structlog.get_logger(__name__)


def run_study(*, n_trials: int = 30, study_name: str = "agrifm_hpo") -> optuna.Study:
    mlflow.set_experiment("hpo_agrifm")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        with mlflow.start_run(nested=True):
            mlflow.log_params({"lr": lr, "weight_decay": weight_decay})
            score = lr * 10.0 + weight_decay
            mlflow.log_metric("proxy_loss", score)
            return score

    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna AgriFM HPO (smoke proxy)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="agrifm_hpo")
    args = parser.parse_args(argv)
    study = run_study(n_trials=args.n_trials, study_name=args.study_name)
    log.info("best_trial", best_value=study.best_value, best_params=study.best_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
