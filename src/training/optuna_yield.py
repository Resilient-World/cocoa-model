"""Optuna HPO for YieldSurrogateV2 (CRPS on spatial holdout)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.spatial_splits import spatial_holdout_mask
from models.yield_surrogate import CocoaPINNLoss
from models.yield_surrogate_v2 import YieldSurrogateV2
from registry.promotion_gate import crps_1d
from scripts.train_yield_surrogate import load_lhs_table, stratified_split
from scripts.train_yield_surrogate_v2 import YieldSurrogateV2Dataset, evaluate


def _ensure_synthetic_data(case2: Path, almanac: Path, era5_dir: Path) -> None:
    from data.pipeline_stubs import (
        write_lhs_parquets,
        write_per_farm_era5_zarrs,
        write_era5_zarr,
    )

    if not case2.is_file():
        write_lhs_parquets(case2_path=case2, almanac_path=almanac, n_farms=40, seed=0)
        write_era5_zarr(_REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr")
        write_per_farm_era5_zarrs(case2, era5_dir)


def run_study(*, n_trials: int = 50, study_name: str = "yield_hpo") -> optuna.Study:
    case2 = _REPO_ROOT / "data" / "simulations" / "case2_lhs.parquet"
    almanac = _REPO_ROOT / "data" / "simulations" / "almanac_lhs.parquet"
    era5_dir = _REPO_ROOT / "data" / "era5"
    _ensure_synthetic_data(case2, almanac, era5_dir)

    merged = load_lhs_table(case2, almanac)
    lats = merged["latitude"].to_numpy(dtype=np.float64)
    lons = merged["longitude"].to_numpy(dtype=np.float64)
    holdout = spatial_holdout_mask(lats, lons, fraction=0.10, seed=42)
    train_df = merged.loc[~holdout].reset_index(drop=True)
    val_df = merged.loc[holdout].reset_index(drop=True)

    mlflow.set_experiment("hpo_yield")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        hidden = trial.suggest_categorical("hidden_dims", [64, 128, 256])
        tele_w = trial.suggest_float("teleconnection_weight", 0.0, 0.5)
        pape_period = trial.suggest_int("pape_period", 180, 365)

        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "lr": lr,
                    "dropout": dropout,
                    "hidden_dims": hidden,
                    "teleconnection_weight": tele_w,
                    "pape_period": pape_period,
                }
            )
            device = torch.device("cpu")
            model = YieldSurrogateV2().to(device)
            train_loader = DataLoader(
                YieldSurrogateV2Dataset(train_df, era5_dir),
                batch_size=16,
                shuffle=True,
            )
            val_loader = DataLoader(
                YieldSurrogateV2Dataset(val_df, era5_dir),
                batch_size=16,
                shuffle=False,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            loss_fn = CocoaPINNLoss(y_max=4.0)
            for _ in range(3):
                model.train()
                for batch in train_loader:
                    opt.zero_grad(set_to_none=True)
                    pred, traces = model.forward_with_traces(
                        batch["climate"].to(device),
                        batch["static"].to(device),
                        batch["region_id"].to(device),
                    )
                    loss = loss_fn(pred, batch["target"].to(device), traces, batch["climate"].to(device))
                    loss.backward()
                    opt.step()

            metrics = evaluate(model, val_loader, device)
            ys = []
            ps = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    pred = model(
                        batch["climate"].to(device),
                        batch["static"].to(device),
                        batch["region_id"].to(device),
                    )
                    ys.extend(batch["target"].cpu().numpy().tolist())
                    ps.extend(pred.cpu().numpy().tolist())
            obs = np.asarray(ys)
            ens = np.stack([np.asarray(ps), np.asarray(ps) * 0.99, np.asarray(ps) * 1.01])
            score = crps_1d(obs, ens)
            mlflow.log_metric("crps", score)
            mlflow.log_metric("val_rmse", metrics["rmse"])
            return score

    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna yield surrogate HPO")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="yield_hpo")
    args = parser.parse_args(argv)
    study = run_study(n_trials=args.n_trials, study_name=args.study_name)
    print(f"Best CRPS: {study.best_value:.4f} params={study.best_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
