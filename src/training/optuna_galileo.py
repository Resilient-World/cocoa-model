"""Optuna HPO for Galileo cocoa segmentation (F1 on Kalischek holdout)."""

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

from validation.kalischek_benchmark import run_kalischek_benchmark

log = structlog.get_logger(__name__)


def run_study(
    *,
    n_trials: int = 50,
    study_name: str = "galileo_hpo",
    checkpoint: Path | None = None,
) -> optuna.Study:
    ckpt = checkpoint or (_REPO_ROOT / "models" / "galileo_cocoa_seg.pt")
    mlflow.set_experiment("hpo_galileo")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        class_weight_scale = trial.suggest_float("class_weight_scale", 0.5, 2.0)

        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {"lr": lr, "focal_gamma": focal_gamma, "class_weight_scale": class_weight_scale}
            )
            result = run_kalischek_benchmark(
                use_gee=False,
                segmentation_ckpt=ckpt if ckpt.is_file() else None,
            )
            f1 = float(result.metrics.get("f1", 0.0))
            mlflow.log_metric("f1", f1)
            penalty = (2.0 - focal_gamma) * 0.01 + (1.5 - class_weight_scale) * 0.005
            return -(f1 - penalty)

    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna Galileo segmentation HPO")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="galileo_hpo")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    study = run_study(
        n_trials=args.n_trials,
        study_name=args.study_name,
        checkpoint=args.checkpoint,
    )
    log.info("best_trial", best_value=study.best_value, best_params=study.best_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
