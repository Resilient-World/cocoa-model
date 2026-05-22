#!/usr/bin/env python3
"""
Fine-tune PAPE + TeleconnectionGNN on top of a frozen YieldSurrogateV2 backbone.

LHS/ALMANAC targets; AdamW lr=1e-4, 30 epochs cosine schedule.
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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.teleconnection_ingest import (
    get_indices_for_year,
    load_indices_table,
    region_key_from_latlon,
)
from models.yield_surrogate import CocoaPINNLoss
from models.yield_surrogate_v2 import YieldSurrogateV2
from models.yield_surrogate_v2_teleconnection import (
    YieldSurrogateV2Teleconnection,
    freeze_surrogate_except_pape,
)
from scripts.train_yield_surrogate_v2 import (
    YieldSurrogateV2Dataset,
    load_lhs_table,
    regression_metrics,
    stratified_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_YEAR = 2023


class TeleconnectionFinetuneDataset(YieldSurrogateV2Dataset):
    """LHS rows with teleconnection index windows."""

    def __init__(
        self,
        table: pd.DataFrame,
        era5_dir: Path,
        *,
        indices_table: pd.DataFrame,
        climate_year: int = TRAIN_YEAR,
        **kwargs: object,
    ) -> None:
        super().__init__(table, era5_dir, **kwargs)  # type: ignore[arg-type]
        self._indices_table = indices_table
        self._climate_year = climate_year

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(idx)
        row = self.table.iloc[idx]
        lat = float(row.get("latitude", 6.0))
        lon = float(row.get("longitude", -2.0))
        region_key = region_key_from_latlon(lat, lon)
        indices = get_indices_for_year(
            self._climate_year,
            region_key,
            table=self._indices_table,
        )
        batch["teleconnection"] = indices
        batch["lat"] = torch.tensor(lat, dtype=torch.float32)
        batch["lon"] = torch.tensor(lon, dtype=torch.float32)
        return batch


def _collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        if key in ("farm_id", "ecozone"):
            out[key] = [b[key] for b in batch]
            continue
        if key == "teleconnection":
            out[key] = {
                "nino34": torch.stack(
                    [torch.from_numpy(b["teleconnection"]["nino34"]) for b in batch]
                ),
                "atl3": torch.stack([torch.from_numpy(b["teleconnection"]["atl3"]) for b in batch]),
                "iod": torch.stack([torch.from_numpy(b["teleconnection"]["iod"]) for b in batch]),
            }
            continue
        out[key] = torch.stack([b[key] for b in batch])
    return out


@torch.no_grad()
def evaluate_composite(
    model: YieldSurrogateV2Teleconnection,
    loader: DataLoader,
    device: torch.device,
    *,
    use_gnn: bool = True,
) -> dict[str, float]:
    model.eval()
    ys: list[float] = []
    ps: list[float] = []
    for batch in loader:
        climate = batch["climate"].to(device)
        static = batch["static"].to(device)
        region_id = batch["region_id"].to(device)
        tele = batch["teleconnection"] if use_gnn else None
        pred = model(
            climate,
            static,
            region_id,
            tele,
            lat=batch["lat"],
            lon=batch["lon"],
        )
        ys.extend(batch["target"].cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
    return regression_metrics(np.asarray(ys), np.asarray(ps))


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    indices_table = load_indices_table(args.teleconnection_parquet)
    merged = load_lhs_table(args.case2_parquet, args.almanac_parquet)
    train_df, val_df, _test_df = stratified_split(merged, seed=args.seed)

    train_ds = TeleconnectionFinetuneDataset(
        train_df,
        args.era5_dir,
        indices_table=indices_table,
        climate_year=args.climate_year,
    )
    val_ds = TeleconnectionFinetuneDataset(
        val_df,
        args.era5_dir,
        indices_table=indices_table,
        climate_year=args.climate_year,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    surrogate = YieldSurrogateV2.from_checkpoint(
        args.surrogate_checkpoint,
        map_location="cpu",
    )
    model = YieldSurrogateV2Teleconnection(surrogate).to(device)
    freeze_surrogate_except_pape(model)
    loss_fn = CocoaPINNLoss(y_max=4.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "climate_year": args.climate_year,
            }
        )

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                region_id = batch["region_id"].to(device)
                target = batch["target"].to(device)
                tele = batch["teleconnection"]

                optimizer.zero_grad(set_to_none=True)
                pred, traces = model.forward_with_traces(
                    climate,
                    static,
                    region_id,
                    tele,
                    lat=batch["lat"],
                    lon=batch["lon"],
                )
                loss = loss_fn(pred, target, traces, climate)

                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            val_metrics = evaluate_composite(model, val_loader, device, use_gnn=True)
            val_no_gnn = evaluate_composite(model, val_loader, device, use_gnn=False)
            mlflow.log_metrics(
                {
                    "val_rmse": val_metrics["rmse"],
                    "val_rmse_no_gnn": val_no_gnn["rmse"],
                },
                step=epoch,
            )
            logger.info(
                "epoch %d/%d loss=%.4f val_rmse=%.4f (no_gnn=%.4f)",
                epoch + 1,
                args.epochs,
                epoch_loss / max(n_batches, 1),
                val_metrics["rmse"],
                val_no_gnn["rmse"],
            )

        checkpoint = {
            "version": "v2_teleconnection",
            "state_dict": model.state_dict(),
            "config": {
                "surrogate_checkpoint": str(args.surrogate_checkpoint),
                "climate_year": args.climate_year,
            },
        }
        torch.save(checkpoint, out_path)
        logger.info("Saved %s", out_path)

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune teleconnection GNN + PAPE")
    p.add_argument(
        "--surrogate-checkpoint",
        type=Path,
        default=_REPO_ROOT / "models" / "yield_surrogate_v2.pt",
    )
    p.add_argument(
        "--teleconnection-parquet",
        type=Path,
        default=_REPO_ROOT / "data" / "external" / "teleconnection_indices.parquet",
    )
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
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "models" / "yield_surrogate_v2_teleconnection.pt",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--climate-year", type=int, default=TRAIN_YEAR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mlflow-experiment", default="resilient-cocoa-yield-teleconnection")
    p.add_argument("--mlflow-run-name", default="teleconnection-finetune")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
