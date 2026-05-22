#!/usr/bin/env python3
"""
Train :class:`models.ensemble_surrogate.CocoaYieldPINN` on ERA5 Zarr + CASE2/ALMANAC LHS runs.

Expects a 1000-farm Latin-hypercube design (``data/simulations/*_lhs.parquet``) paired with
per-farm ERA5 Zarr stores under ``data/era5/*.zarr``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import xarray as xr

from data.era5_ingest import compute_derived_features
from models.ensemble_surrogate import (
    CLIMATE_FEATURE_NAMES,
    N_CLIMATE,
    N_STATIC,
    SEQ_LEN,
    CocoaYieldPINN,
    YieldEnsemble,
    physics_residual_loss,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ERA5 variable names in Zarr → PINN climate channel order
_ERA5_CLIMATE_MAP: dict[str, str] = {
    "tmean": "tmean",
    "tmax": "tmax",
    "tmin": "tmin",
    "vpd": "vpd_mean",
    "et0": "et0",
    "cwd": "cwd",
    "sm_root": "sm_root",
    "precip": "precip",
    "srad": "srad",
    "gdd_cocoa": "gdd_cocoa",
    "heat_days_32c": "heat_days_above_32c",
    "dry_spell_max": "dry_spell_max",
}


class TrainCocoaYieldPINN(CocoaYieldPINN):
    """AdamW + cosine LR schedule for surrogate training."""

    def __init__(
        self,
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class FarmYieldDataset(Dataset[dict[str, Tensor]]):
    """One sample per LHS farm: ERA5 climate window + static features + dual targets."""

    def __init__(
        self,
        table: pd.DataFrame,
        era5_dir: Path,
        *,
        climate_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.era5_dir = era5_dir
        self._climate_cache = climate_cache if climate_cache is not None else {}

    def __len__(self) -> int:
        return len(self.table)

    def _climate_for_farm(self, farm_id: str) -> np.ndarray:
        if farm_id in self._climate_cache:
            return self._climate_cache[farm_id]
        zarr_path = self.era5_dir / f"{farm_id}.zarr"
        if not zarr_path.is_dir():
            matches = list(self.era5_dir.glob(f"{farm_id}*.zarr"))
            if not matches:
                raise FileNotFoundError(
                    f"No ERA5 Zarr for farm_id={farm_id!r} under {self.era5_dir}"
                )
            zarr_path = matches[0]
        tensor = climate_tensor_from_zarr(xr.open_zarr(zarr_path, consolidated=True))
        self._climate_cache[farm_id] = tensor
        return tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self.table.iloc[idx]
        farm_id = str(row["farm_id"])
        climate = self._climate_for_farm(farm_id).astype(np.float32)
        static = np.array(
            [
                row["planting_density"],
                row["tree_age"],
                row["slai"],
                row["soil_fc"],
                row["soil_wp"],
                row["soil_depth"],
                row["elevation"],
                row["latitude"],
            ],
            dtype=np.float32,
        )
        return {
            "X_climate": torch.from_numpy(climate),
            "X_static": torch.from_numpy(static),
            "y_case2": torch.tensor(float(np.log1p(row["y_case2"])), dtype=torch.float32),
            "y_almanac": torch.tensor(float(np.log1p(row["y_almanac"])), dtype=torch.float32),
            "farm_id": farm_id,
            "ecozone": str(row["ecozone"]),
        }


def climate_tensor_from_zarr(ds: xr.Dataset) -> np.ndarray:
    """Build ``[n_climate, seq_len]`` tensor from an ERA5-Land Zarr dataset."""
    if "time" not in ds.dims:
        raise ValueError("ERA5 Zarr must include a time dimension")

    needed = {"tmean", "tmax", "tmin", "precip", "srad", "vpd_mean", "et0", "cwd", "sm_root"}
    if not needed.issubset(set(ds.data_vars)):
        missing = needed - set(ds.data_vars)
        raise ValueError(f"ERA5 Zarr missing variables: {sorted(missing)}")

    ds = compute_derived_features(ds)
    ds = ds.sortby("time").isel(time=slice(-SEQ_LEN, None))
    if ds.sizes.get("time", 0) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} daily timesteps, got {ds.sizes.get('time')}")

    channels: list[np.ndarray] = []
    for name in CLIMATE_FEATURE_NAMES:
        era5_var = _ERA5_CLIMATE_MAP[name]
        if era5_var not in ds:
            raise KeyError(f"Derived variable {era5_var!r} not in dataset for channel {name!r}")
        values = np.asarray(ds[era5_var].values, dtype=np.float64).reshape(-1)
        if values.size == 1:
            values = np.full(SEQ_LEN, float(values))
        elif values.size < SEQ_LEN:
            pad = np.full(SEQ_LEN - values.size, values[0])
            values = np.concatenate([pad, values])
        else:
            values = values[-SEQ_LEN:]
        channels.append(values)
    return np.stack(channels, axis=0)


def load_lhs_table(case2_path: Path, almanac_path: Path) -> pd.DataFrame:
    """Merge CASE2 and ALMANAC LHS simulation tables on ``farm_id``."""
    case2 = pd.read_parquet(case2_path)
    almanac = pd.read_parquet(almanac_path)

    for col in ("farm_id", "ecozone", "planting_density", "tree_age", "slai", "soil_fc"):
        if col not in case2.columns:
            raise ValueError(f"{case2_path} missing column {col!r}")

    y_case2_col = "y_case2" if "y_case2" in case2.columns else "yearly_yield_kg_ha"
    y_alm_col = "y_almanac" if "y_almanac" in almanac.columns else "yearly_yield_kg_ha"

    merged = case2.merge(
        almanac[["farm_id", y_alm_col]].rename(columns={y_alm_col: "y_almanac"}),
        on="farm_id",
        how="inner",
    ).rename(columns={y_case2_col: "y_case2"})

    defaults = {
        "soil_wp": 0.14,
        "soil_depth": 150.0,
        "elevation": 200.0,
        "latitude": 6.0,
        "longitude": -2.0,
    }
    if "tree_age" not in merged.columns and "tree_age_years" in merged.columns:
        merged["tree_age"] = merged["tree_age_years"]

    for key, val in defaults.items():
        if key not in merged.columns:
            merged[key] = val

    return merged


def stratified_split(
    df: pd.DataFrame,
    *,
    ecozone_col: str = "ecozone",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 split stratified by ecozone."""
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df[ecozone_col],
        random_state=seed,
    )
    rel_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - rel_val),
        stratify=temp_df[ecozone_col],
        random_state=seed,
    )
    return train_df, val_df, test_df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@torch.no_grad()
def collect_predictions(
    model: CocoaYieldPINN,
    loader: DataLoader[dict[str, Tensor]],
) -> dict[str, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    y_case2: list[float] = []
    y_almanac: list[float] = []
    ecozones: list[str] = []
    for batch in loader:
        out = model(batch["X_climate"], batch["X_static"]).cpu().numpy()
        preds.append(out)
        y_case2.extend(batch["y_case2"].cpu().numpy().tolist())
        y_almanac.extend(batch["y_almanac"].cpu().numpy().tolist())
        ecozones.extend(batch["ecozone"])
    stacked = np.concatenate(preds, axis=0)
    return {
        "pred_log": stacked,
        "y_case2_log": np.asarray(y_case2),
        "y_almanac_log": np.asarray(y_almanac),
        "ecozone": np.asarray(ecozones, dtype=object),
    }


class MLflowMetricsCallback(pl.Callback):
    """Log RMSE/MAE/R², physics loss, and Love-plot calibration each validation epoch."""

    def __init__(self, val_loader: DataLoader[dict[str, Tensor]]) -> None:
        super().__init__()
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: CocoaYieldPINN) -> None:
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        pl_module.eval()
        all_pred: list[np.ndarray] = []
        all_case2: list[np.ndarray] = []
        all_alm: list[np.ndarray] = []
        phys_losses: list[float] = []

        for batch in self.val_loader:
            x_c = batch["X_climate"]
            x_s = batch["X_static"]
            climate_grad = x_c.detach().clone().requires_grad_(True)
            pred = pl_module(climate_grad, x_s)
            phys = physics_residual_loss(pred, climate_grad, lambda_phys=pl_module.lambda_phys)
            phys_losses.append(float(phys.detach().cpu()))

            all_pred.append(pred.detach().cpu().numpy())
            all_case2.append(batch["y_case2"].cpu().numpy())
            all_alm.append(batch["y_almanac"].cpu().numpy())

        pred_log = np.concatenate(all_pred, axis=0)
        true_case2_log = np.concatenate(all_case2, axis=0)
        true_alm_log = np.concatenate(all_alm, axis=0)

        pred_case2_kg = np.expm1(pred_log[:, 0])
        pred_alm_kg = np.expm1(pred_log[:, 1])
        true_case2_kg = np.expm1(true_case2_log)
        true_alm_kg = np.expm1(true_alm_log)

        m_case2 = regression_metrics(true_case2_kg, pred_case2_kg)
        m_alm = regression_metrics(true_alm_kg, pred_alm_kg)
        m_mean = regression_metrics(
            np.concatenate([true_case2_kg, true_alm_kg]),
            np.concatenate([pred_case2_kg, pred_alm_kg]),
        )

        mlflow.log_metric("val_phys_loss", float(np.mean(phys_losses)), step=epoch)
        for prefix, metrics in (
            ("val_case2", m_case2),
            ("val_almanac", m_alm),
            ("val_mean", m_mean),
        ):
            for name, value in metrics.items():
                mlflow.log_metric(f"{prefix}_{name}", value, step=epoch)

        self._log_love_plot(true_case2_kg, pred_case2_kg, "case2", epoch)
        self._log_love_plot(true_alm_kg, pred_alm_kg, "almanac", epoch)

    @staticmethod
    def _log_love_plot(y_true: np.ndarray, y_pred: np.ndarray, tag: str, epoch: int) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping Love plot for %s", tag)
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xlabel("Observed yield (kg/ha)")
        ax.set_ylabel("Predicted yield (kg/ha)")
        ax.set_title(f"Love plot — {tag} (epoch {epoch})")
        ax.legend(loc="upper left")
        fig.tight_layout()
        mlflow.log_figure(fig, f"love_plot_{tag}_epoch_{epoch:03d}.png")
        plt.close(fig)


def save_stacking_weights(
    ensemble: YieldEnsemble,
    path: Path,
) -> None:
    payload = {
        eco: {"w_case2": float(w[0]), "w_almanac": float(w[1])}
        for eco, w in ensemble._weights.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fit_stacking_from_predictions(
    pred_log: np.ndarray,
    y_case2_log: np.ndarray,
    y_almanac_log: np.ndarray,
    ecozones: np.ndarray,
) -> YieldEnsemble:
    """Build OOF-style stacking table from validation predictions."""
    df = pd.DataFrame(
        {
            "ecozone": ecozones,
            "y_true": 0.5 * (y_case2_log + y_almanac_log),
            "pinn_case2": pred_log[:, 0],
            "pinn_almanac": pred_log[:, 1],
        }
    )
    ens = YieldEnsemble()
    ens.fit_stacking(df)
    return ens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CocoaYieldPINN yield surrogate")
    parser.add_argument(
        "--case2-parquet", type=Path, default=_REPO_ROOT / "data/simulations/case2_lhs.parquet"
    )
    parser.add_argument(
        "--almanac-parquet", type=Path, default=_REPO_ROOT / "data/simulations/almanac_lhs.parquet"
    )
    parser.add_argument("--era5-dir", type=Path, default=_REPO_ROOT / "data/era5")
    parser.add_argument("--checkpoint", type=Path, default=_REPO_ROOT / "models/pinn_v1.ckpt")
    parser.add_argument(
        "--ensemble-weights", type=Path, default=_REPO_ROOT / "models/ensemble_weights.json"
    )
    parser.add_argument(
        "--metrics-json", type=Path, default=_REPO_ROOT / "reports/pinn_metrics.json"
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-phys", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-experiment", type=str, default="resilient-cocoa-yield-pinn")
    parser.add_argument("--mlflow-run-name", type=str, default="pinn_v1")
    args = parser.parse_args(argv)

    pl.seed_everything(args.seed, workers=True)

    table = load_lhs_table(args.case2_parquet, args.almanac_parquet)
    logger.info("Loaded %d farms from LHS parquets", len(table))

    train_df, val_df, test_df = stratified_split(table, seed=args.seed)
    logger.info("Split sizes train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    climate_cache: dict[str, np.ndarray] = {}
    train_ds = FarmYieldDataset(train_df, args.era5_dir, climate_cache=climate_cache)
    val_ds = FarmYieldDataset(val_df, args.era5_dir, climate_cache=climate_cache)
    test_ds = FarmYieldDataset(test_df, args.era5_dir, climate_cache=climate_cache)

    def _collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor | list[str]]:
        return {
            "X_climate": torch.stack([b["X_climate"] for b in batch]),
            "X_static": torch.stack([b["X_static"] for b in batch]),
            "y_case2": torch.stack([b["y_case2"] for b in batch]),
            "y_almanac": torch.stack([b["y_almanac"] for b in batch]),
            "farm_id": [b["farm_id"] for b in batch],
            "ecozone": [b["ecozone"] for b in batch],
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    model = TrainCocoaYieldPINN(
        max_epochs=args.max_epochs,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint.parent,
        filename=args.checkpoint.stem,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")
    mlflow_cb = MLflowMetricsCallback(val_loader)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_phys": args.lambda_phys,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_climate": N_CLIMATE,
                "n_static": N_STATIC,
            }
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            callbacks=[checkpoint_cb, early_stop, mlflow_cb],
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
        trainer.fit(model, train_loader, val_loader)

        best_path = Path(checkpoint_cb.best_model_path or "")
        if best_path.is_file():
            if best_path.resolve() != args.checkpoint.resolve():
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_path, args.checkpoint)
            best_path = args.checkpoint
            model = TrainCocoaYieldPINN.load_from_checkpoint(str(best_path))
        else:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(args.checkpoint)
            best_path = args.checkpoint

        val_preds = collect_predictions(model, val_loader)
        ensemble = fit_stacking_from_predictions(
            val_preds["pred_log"],
            val_preds["y_case2_log"],
            val_preds["y_almanac_log"],
            val_preds["ecozone"],
        )
        save_stacking_weights(ensemble, args.ensemble_weights)

        test_preds = collect_predictions(model, test_loader)
        pred_kg = np.expm1(test_preds["pred_log"])
        metrics = {
            "checkpoint": str(best_path),
            "test_case2": regression_metrics(
                np.expm1(test_preds["y_case2_log"]),
                pred_kg[:, 0],
            ),
            "test_almanac": regression_metrics(
                np.expm1(test_preds["y_almanac_log"]),
                pred_kg[:, 1],
            ),
        }
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(args.metrics_json))
        mlflow.log_artifact(str(args.ensemble_weights))

        logger.info("Best checkpoint: %s", best_path)
        logger.info("Ensemble weights: %s", args.ensemble_weights)
        logger.info("Metrics: %s", args.metrics_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
