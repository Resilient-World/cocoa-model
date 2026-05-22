#!/usr/bin/env python3
"""
Train :class:`models.casej_surrogate.CASEJSurrogate` on CASEJ LHS parquet with PINN losses.

Validates on held-out CASEJ runs and ICCO/FAO country yields (GHA, CIV 2015–2023).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.casej_surrogate import (
    DEFAULT_CASEJ_CHECKPOINT,
    CASEJPhysicsLoss,
    CASEJSurrogate,
)
from validation.icco_yield_backtest import load_icco_table

logger = logging.getLogger(__name__)
DEFAULT_DATA = _REPO_ROOT / "data" / "simulations" / "casej_lhs.parquet"


class CASEJParquetDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._df.iloc[idx]
        climate = np.frombuffer(row["climate"], dtype=np.float32).reshape(365, 11).copy()
        static = np.frombuffer(row["static"], dtype=np.float32).copy()
        return {
            "climate": torch.from_numpy(climate),
            "static": torch.from_numpy(static),
            "co2_ppm": torch.tensor([float(row["co2_ppm"])], dtype=torch.float32),
            "yield": torch.tensor([float(row["yield_t_ha"])], dtype=torch.float32),
        }


def _icco_country_validation(
    model: CASEJSurrogate,
    device: torch.device,
    *,
    countries: tuple[str, ...] = ("GHA", "CIV"),
    years: tuple[int, ...] = tuple(range(2015, 2024)),
) -> dict[str, float]:
    """Light ICCO sanity check: correlate model vs observed national yield (t/ha)."""
    try:
        icco = load_icco_table()
    except FileNotFoundError:
        logger.warning("ICCO table missing; skipping country validation")
        return {"icco_r2": float("nan"), "icco_mae": float("nan")}

    icco = icco[icco["country_iso3"].isin(countries) & icco["year"].isin(years)]
    if icco.empty:
        return {"icco_r2": float("nan"), "icco_mae": float("nan")}

    from models.casej_process import (
        CASEJSite,
        site_to_static_vector,
        synthesize_daily_weather,
        weather_to_climate_tensor,
    )

    preds: list[float] = []
    obs: list[float] = []
    model.eval()
    for _, row in icco.iterrows():
        iso = str(row["country_iso3"])
        year = int(row["year"])
        area = float(row["planted_area_ha"])
        obs_y = float(row["production_tonnes"]) / area
        lat = 6.0 if iso == "GHA" else 5.8
        lon = -1.2 if iso == "GHA" else -5.3
        co2 = 400.0 + 2.2 * (year - 2015)
        weather = synthesize_daily_weather(365, seed=year + hash(iso) % 1000)
        site = CASEJSite(lat=lat, lon=lon, awc_mm=150.0, slai=1.0, tree_age_y=12.0, co2_ppm=co2)
        static = torch.from_numpy(site_to_static_vector(site)).unsqueeze(0).to(device)
        climate = torch.from_numpy(weather_to_climate_tensor(weather, co2)).unsqueeze(0).to(device)
        co2_t = torch.tensor([co2], device=device)
        with torch.no_grad():
            pred = float(model(climate, static, co2_ppm=co2_t).item())
        preds.append(pred)
        obs.append(obs_y)

    return {
        "icco_r2": float(r2_score(obs, preds)),
        "icco_mae": float(mean_absolute_error(obs, preds)),
    }


def train(
    df: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, float]:
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    train_loader = DataLoader(CASEJParquetDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CASEJParquetDataset(val_df), batch_size=batch_size)

    model = CASEJSurrogate().to(device)
    loss_fn = CASEJPhysicsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mlflow.set_experiment("casej_surrogate")
    with mlflow.start_run(run_name="casej_pinn"):
        mlflow.log_params({"epochs": epochs, "n_train": len(train_df), "n_val": len(val_df)})
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                co2 = batch["co2_ppm"].squeeze(-1).to(device)
                y = batch["yield"].squeeze(-1).to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(climate, static, co2_ppm=co2)
                loss = loss_fn(pred, y, model, climate, static)
                loss.backward()
                opt.step()
                train_loss += float(loss.item())
                n_batches += 1
            if epoch % 5 == 0 or epoch == epochs - 1:
                mlflow.log_metric("train_loss", train_loss / max(n_batches, 1), step=epoch)

        model.eval()
        val_preds: list[float] = []
        val_obs: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                co2 = batch["co2_ppm"].squeeze(-1).to(device)
                y = batch["yield"].squeeze(-1).to(device)
                pred = model(climate, static, co2_ppm=co2)
                val_preds.extend(pred.cpu().numpy().tolist())
                val_obs.extend(y.cpu().numpy().tolist())
        val_mae = mean_absolute_error(val_obs, val_preds)
        val_r2 = r2_score(val_obs, val_preds)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)

        icco_metrics = _icco_country_validation(model, device)
        for k, v in icco_metrics.items():
            mlflow.log_metric(k, v)

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "val_mae": val_mae, "val_r2": val_r2}, checkpoint
        )
        mlflow.log_artifact(str(checkpoint))

    logger.info(
        "Saved CASEJ surrogate to %s (val_mae=%.3f, val_r2=%.3f)", checkpoint, val_mae, val_r2
    )
    return {"val_mae": val_mae, "val_r2": val_r2, **icco_metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CASEJ PINN surrogate")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CASEJ_CHECKPOINT)
    parser.add_argument(
        "--generate", action="store_true", help="Run LHS generator if parquet missing"
    )
    parser.add_argument("--n-generate", type=int, default=1500)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.data.is_file():
        if args.generate:
            import importlib.util

            gen_path = _REPO_ROOT / "scripts" / "generate_casej_training_set.py"
            spec = importlib.util.spec_from_file_location("generate_casej_training_set", gen_path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            args.data.parent.mkdir(parents=True, exist_ok=True)
            mod.generate_lhs(
                args.n_generate,
                seed=42,
                params_path=_REPO_ROOT / "config" / "casej" / "params.yaml",
            ).to_parquet(args.data)
        else:
            raise FileNotFoundError(f"Training parquet not found: {args.data}. Run with --generate")

    df = pd.read_parquet(args.data)
    train(
        df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
