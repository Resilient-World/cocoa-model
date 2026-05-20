"""
Calibrate :class:`~models.yield_surrogate.YieldSurrogateModel` on ICCO + CRIG yield panel.

Hydra entrypoint::

    python -m training.train_yield --config-name yield max_epochs=2
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data.yield_panel import (
    PanelRow,
    YieldPanelDataset,
    build_yield_panel,
    panel_train_val_split,
)
from models.yield_surrogate import CocoaPINNLoss, YieldSurrogateModel

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config" / "training"


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


@torch.no_grad()
def _evaluate(
    model: YieldSurrogateModel,
    rows: list[PanelRow],
    loss_fn: CocoaPINNLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_country: dict[str, list[float]] = {}
    per_cohort: dict[str, list[float]] = {}
    sq_err: list[float] = []
    for row in rows:
        climate = torch.from_numpy(row.climate).unsqueeze(0).to(device)
        static = torch.from_numpy(row.static).unsqueeze(0).to(device)
        pred, traces = model.forward_with_traces(climate, static)
        target = torch.tensor([row.yield_target_pre_biotic_t_ha], device=device)
        err = float((pred - target).pow(2).item())
        sq_err.append(err)
        per_country.setdefault(row.country_iso3, []).append(err)
        per_cohort.setdefault(row.cohort, []).append(err)
    country_rmse = {iso: float(np.sqrt(np.mean(v))) for iso, v in per_country.items()}
    cohort_rmse = {c: float(np.sqrt(np.mean(v))) for c, v in per_cohort.items()}
    return {
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "country_rmse": country_rmse,
        "cohort_rmse": cohort_rmse,
    }


def train_yield_surrogate(cfg: DictConfig) -> Path:
    """Run AdamW + cosine calibration; save checkpoint and log MLflow metrics."""
    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")

    panel_rows = build_yield_panel(
        icco_glob=_resolve_path(cfg.icco_glob),
        crig_path=_resolve_path(cfg.crig_path),
        bootstrap_per_country_year=int(cfg.bootstrap_per_country_year),
        augment_sigma_t_ha=float(cfg.augment_sigma_t_ha),
        sequence_length=int(cfg.sequence_length),
        seed=int(cfg.seed),
    )
    train_rows, val_rows = panel_train_val_split(panel_rows, val_fraction=float(cfg.val_fraction))

    train_loader = DataLoader(
        YieldPanelDataset(train_rows),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )

    model = YieldSurrogateModel(sequence_length=int(cfg.sequence_length)).to(device)
    loss_fn = CocoaPINNLoss(y_max=4.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.max_epochs),
    )

    checkpoint_path = _resolve_path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(str(cfg.mlflow_experiment))
    with mlflow.start_run(run_name=str(cfg.mlflow_run_name)):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        for epoch in range(int(cfg.max_epochs)):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                climate = batch["climate"].to(device)
                static = batch["static"].to(device)
                target = batch["target"].to(device)

                optimizer.zero_grad(set_to_none=True)
                pred, traces = model.forward_with_traces(climate, static)
                loss = loss_fn(pred, target, traces, climate)
                if isinstance(loss, dict):
                    loss = loss["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            mean_loss = epoch_loss / max(n_batches, 1)

            if epoch % int(cfg.log_every_n_epochs) == 0 or epoch == int(cfg.max_epochs) - 1:
                val_metrics = _evaluate(model, val_rows, loss_fn, device)
                mlflow.log_metric("train_loss", mean_loss, step=epoch)
                mlflow.log_metric("val_rmse", val_metrics["rmse"], step=epoch)
                mlflow.log_metric(
                    "rue",
                    float(model.mechanistic.rue.detach().cpu()),
                    step=epoch,
                )
                mlflow.log_metric(
                    "harvest_index",
                    float(model.mechanistic.harvest_index.detach().cpu()),
                    step=epoch,
                )
                for iso, rmse in val_metrics["country_rmse"].items():
                    mlflow.log_metric(f"rmse_{iso}", rmse, step=epoch)
                for cohort, rmse in val_metrics["cohort_rmse"].items():
                    mlflow.log_metric(f"rmse_{cohort}", rmse, step=epoch)
                logger.info(
                    "epoch %d/%d loss=%.4f val_rmse=%.4f",
                    epoch + 1,
                    cfg.max_epochs,
                    mean_loss,
                    val_metrics["rmse"],
                )

        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        try:
            mlflow.pytorch.log_model(model, artifact_path="yield_surrogate")
        except Exception as exc:  # noqa: BLE001 — optional MLflow flavor
            logger.warning("MLflow pytorch log_model skipped: %s", exc)

        final_val = _evaluate(model, val_rows, loss_fn, device)
        mlflow.log_metric("final_val_rmse", final_val["rmse"])

    logger.info("Saved yield checkpoint to %s", checkpoint_path)
    return checkpoint_path


def _strip_hydra_cli_flags(argv: list[str]) -> tuple[str, list[str]]:
    """Drop ``--config-name`` (handled by compose) and return config name + overrides."""
    config_name = "yield"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-name":
            if i + 1 < len(argv):
                config_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        overrides.append(arg)
        i += 1
    return config_name, overrides


def main(argv: list[str] | None = None) -> int:
    """Hydra CLI: ``python -m training.train_yield --config-name yield max_epochs=2``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    raw = list(argv) if argv is not None else sys.argv[1:]
    config_name, overrides = _strip_hydra_cli_flags(raw)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    train_yield_surrogate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
