#!/usr/bin/env python3
"""
Train :class:`models.yield_surrogate_v2.YieldSurrogateV2` on ERA5 Zarr + CASE2/ALMANAC LHS.

Maps 12-channel PINN climate tensors to the 11-channel yield surrogate layout and
13-d static site features; passes ``region_id`` into PAPE during training.
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
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.era5_ingest import compute_derived_features
from models.pape import region_to_id
from models.yield_surrogate import (
    CLIMATE_CHANNEL_NAMES,
    CLIMATE_IDX,
    STATIC_FEATURE_NAMES,
    STATIC_IDX,
    CocoaPINNLoss,
    pack_tree_age_static,
)
from models.yield_surrogate_v2 import YieldSurrogateV2, region_id_from_latlon
from scripts.train_yield_surrogate import (
    climate_tensor_from_zarr,
    load_lhs_table,
    stratified_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEQ_LEN = 365
DEFAULT_CO2_PPM = 415.0
DEFAULT_WIND10M = 2.0
DEFAULT_RH_MEAN = 75.0

_PINN_TO_SURROGATE: dict[str, str] = {
    "tmax": "tmax",
    "tmin": "tmin",
    "tmean": "tmean",
    "precip": "precip",
    "srad": "srad",
    "vpd": "vpd",
    "et0": "et0",
    "sm_root": "sm_root",
}


def pinn_climate_to_surrogate(pinn_tensor: np.ndarray, *, year: int = 2023) -> np.ndarray:
    """
    Map ``[12, T]`` PINN ERA5 stack to ``[T, 11]`` yield surrogate channels.
    """
    from models.ensemble_surrogate import CLIMATE_FEATURE_NAMES as PINN_NAMES

    t = pinn_tensor.shape[1]
    out = np.zeros((t, len(CLIMATE_CHANNEL_NAMES)), dtype=np.float32)
    for pinn_name, sur_name in _PINN_TO_SURROGATE.items():
        pi = PINN_NAMES.index(pinn_name)
        si = CLIMATE_IDX[sur_name]
        out[:, si] = pinn_tensor[pi].astype(np.float32)
    out[:, CLIMATE_IDX["wind10m"]] = DEFAULT_WIND10M
    out[:, CLIMATE_IDX["rh_mean"]] = DEFAULT_RH_MEAN
    co2 = DEFAULT_CO2_PPM + max(0, year - 2020) * 2.5
    out[:, CLIMATE_IDX["co2_ppm"]] = co2
    return out


def lhs_static_to_surrogate(row: pd.Series) -> np.ndarray:
    """Map LHS soil/management row to 13-d site static vector."""
    fc = float(row.get("soil_fc", 0.35))
    wp = float(row.get("soil_wp", 0.14))
    depth_cm = float(row.get("soil_depth", 150.0))
    awc_mm = max(50.0, (fc - wp) * depth_cm * 10.0)
    tree_age = float(row.get("tree_age", row.get("tree_age_years", 12.0)))
    density = float(row.get("planting_density", 1100.0))
    lat = float(row.get("latitude", 6.0))

    static = np.zeros(len(STATIC_FEATURE_NAMES), dtype=np.float32)
    static[STATIC_IDX["awc_mm"]] = awc_mm
    static[STATIC_IDX["sand_frac"]] = 0.35
    static[STATIC_IDX["baseline_yield_scaled"]] = 0.0
    static[STATIC_IDX["intervention_flag"]] = 0.0
    static[STATIC_IDX["stress_tolerance_flag"]] = 0.0
    static[STATIC_IDX["clay_frac"]] = 0.25
    static[STATIC_IDX["soc_norm"]] = 0.5
    static[STATIC_IDX["ph_norm"]] = 0.55
    static[STATIC_IDX["treecover_norm"]] = 0.4
    static[STATIC_IDX["cocoa_prob"]] = 0.85
    age_norm, cohort, dens_norm = pack_tree_age_static(tree_age, planting_density_trees_ha=density)
    static[STATIC_IDX["tree_age_years_norm"]] = age_norm
    static[STATIC_IDX["cohort_phase"]] = cohort
    static[STATIC_IDX["planting_density_norm"]] = dens_norm
    _ = lat
    return static


def ecozone_to_region_id(ecozone: str, lat: float, lon: float) -> int:
    """Heuristic region id from LHS ecozone label and coordinates."""
    eco = str(ecozone).strip().lower()
    if eco in ("humidforest", "semi_deciduous", "semi-deciduous"):
        return region_to_id("ghana" if lat >= 6.0 else "civ")
    if eco in ("forest", "derived_savanna"):
        return region_to_id("cameroon" if lon > 5.0 else "nigeria")
    return region_id_from_latlon(lat, lon)


class YieldSurrogateV2Dataset(Dataset[dict[str, Tensor]]):
    """LHS farm rows adapted for YieldSurrogateV2 training."""

    def __init__(
        self,
        table: pd.DataFrame,
        era5_dir: Path,
        *,
        climate_cache: dict[str, np.ndarray] | None = None,
        target_col: str = "y_almanac",
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.era5_dir = era5_dir
        self._climate_cache = climate_cache if climate_cache is not None else {}
        self.target_col = target_col
        self._pinn_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.table)

    def _pinn_climate(self, farm_id: str) -> np.ndarray:
        if farm_id in self._pinn_cache:
            return self._pinn_cache[farm_id]
        zarr_path = self.era5_dir / f"{farm_id}.zarr"
        if not zarr_path.is_dir():
            matches = list(self.era5_dir.glob(f"{farm_id}*.zarr"))
            if not matches:
                raise FileNotFoundError(
                    f"No ERA5 Zarr for farm_id={farm_id!r} under {self.era5_dir}"
                )
            zarr_path = matches[0]
        ds = xr.open_zarr(zarr_path, consolidated=True)
        if "gdd_cocoa" not in ds.data_vars:
            ds = compute_derived_features(ds)
        tensor = climate_tensor_from_zarr(ds)
        self._pinn_cache[farm_id] = tensor
        return tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self.table.iloc[idx]
        farm_id = str(row["farm_id"])
        pinn = self._pinn_climate(farm_id)
        climate = pinn_climate_to_surrogate(pinn)
        static = lhs_static_to_surrogate(row)
        lat = float(row.get("latitude", 6.0))
        lon = float(row.get("longitude", -2.0))
        region_id = ecozone_to_region_id(str(row["ecozone"]), lat, lon)

        y_raw = float(row[self.target_col])
        if y_raw > 20.0:
            y_t_ha = y_raw / 1000.0
        else:
            y_t_ha = float(np.expm1(np.log1p(y_raw)))

        return {
            "climate": torch.from_numpy(climate),
            "static": torch.from_numpy(static),
            "region_id": torch.tensor(region_id, dtype=torch.long),
            "target": torch.tensor(y_t_ha, dtype=torch.float32),
            "farm_id": farm_id,
            "ecozone": str(row["ecozone"]),
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@torch.no_grad()
def evaluate(
    model: YieldSurrogateV2,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ys: list[float] = []
    ps: list[float] = []
    for batch in loader:
        pred = model(
            batch["climate"].to(device),
            batch["static"].to(device),
            batch["region_id"].to(device),
        )
        ys.extend(batch["target"].cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
    return regression_metrics(np.asarray(ys), np.asarray(ps))


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.synthetic:
        from data.pipeline_stubs import (
            write_era5_zarr,
            write_lhs_parquets,
            write_per_farm_era5_zarrs,
        )

        write_lhs_parquets(
            n_farms=args.n_synthetic_farms,
            seed=args.seed,
            case2_path=args.case2_parquet,
            almanac_path=args.almanac_parquet,
        )
        zarr_path = _REPO_ROOT / "data" / "processed" / "era5_2020_2024.zarr"
        write_era5_zarr(zarr_path)
        write_per_farm_era5_zarrs(args.case2_parquet, args.era5_dir)

    merged = load_lhs_table(args.case2_parquet, args.almanac_parquet)
    train_df, val_df, test_df = stratified_split(merged, seed=args.seed)

    train_ds = YieldSurrogateV2Dataset(train_df, args.era5_dir)
    val_ds = YieldSurrogateV2Dataset(val_df, args.era5_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = YieldSurrogateV2().to(device)
    loss_fn = CocoaPINNLoss(y_max=4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phenology_yaml = _REPO_ROOT / "config" / "phenology.yaml"

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }
        )

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                region_id = batch["region_id"].to(device)
                target = batch["target"].to(device)

                optimizer.zero_grad(set_to_none=True)
                pred, traces = model.forward_with_traces(climate, static, region_id)
                loss = loss_fn(pred, target, traces, climate)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch,
            )
            logger.info(
                "epoch %d/%d train_loss=%.4f val_rmse=%.4f",
                epoch + 1,
                args.epochs,
                train_loss / max(n_batches, 1),
                val_metrics["rmse"],
            )

        test_loader = DataLoader(
            YieldSurrogateV2Dataset(test_df, args.era5_dir),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_metrics = evaluate(model, test_loader, device)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        checkpoint = {
            "version": "v2",
            "state_dict": model.state_dict(),
            "config": {
                "sequence_length": model.sequence_length,
                "climate_features": model.climate_features,
                "galileo_dim": model.galileo_dim,
                "phenology_yaml": str(phenology_yaml),
            },
        }
        torch.save(checkpoint, out_path)
        logger.info("Saved %s (test_rmse=%.4f)", out_path, test_metrics["rmse"])
        if args.metrics_out:
            from data.pipeline_stubs import write_metrics_json

            write_metrics_json(
                args.metrics_out,
                {**{f"test_{k}": v for k, v in test_metrics.items()}, "epochs": args.epochs},
            )

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YieldSurrogateV2 on LHS/ALMANAC data")
    p.add_argument(
        "--case2-parquet",
        type=Path,
        default=_REPO_ROOT / "data" / "simulations" / "case2_lhs.parquet",
    )
    p.add_argument(
        "--almanac-parquet",
        type=Path,
        default=_REPO_ROOT / "data" / "simulations" / "almanac_lhs.parquet",
    )
    p.add_argument("--era5-dir", type=Path, default=_REPO_ROOT / "data" / "era5")
    p.add_argument("--output", type=Path, default=_REPO_ROOT / "models" / "yield_surrogate_v2.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mlflow-experiment", default="resilient-cocoa-yield-v2")
    p.add_argument("--mlflow-run-name", default="yield-surrogate-v2")
    p.add_argument("--synthetic", action="store_true", help="Generate LHS + ERA5 stubs")
    p.add_argument("--n-synthetic-farms", type=int, default=60)
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="JSON metrics for DVC (e.g. reports/train/yield_v2_metrics.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
