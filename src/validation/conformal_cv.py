"""Blocked conformal coverage evaluation across CV strategies."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from data.yield_panel import COUNTRY_CENTROIDS, PanelRow, build_yield_panel
from models.conformal.cqr import ConformalCalibrator, QuantileYieldSurrogate, pinball_loss
from validation.spatial_cv import BufferedLOO, SpatialBlockSplit
from validation.temporal_cv import ForwardChainSplit

CvStrategy = Literal["random", "spatial_block", "temporal_forward", "buffered_loo"]

COVERAGE_LO = 0.88
COVERAGE_HI = 0.92
NOMINAL = 0.90


def _interval_pit(y: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
    """PIT values from fixed prediction intervals (uniform under correct calibration)."""
    width = np.maximum(uppers - lowers, 1e-6)
    pit = (y - lowers) / width
    return np.clip(pit, 0.0, 1.0)


def _panel_coords(rows: list[PanelRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = []
    lons = []
    years = []
    for i, r in enumerate(rows):
        centroid = COUNTRY_CENTROIDS.get(r.country_iso3, (6.0, -2.0))
        # Deterministic jitter so spatial blocks are not collapsed to country centroids.
        lats.append(centroid[0] + 0.02 * ((i * 7) % 41 - 20))
        lons.append(centroid[1] + 0.02 * ((i * 13) % 37 - 18))
        years.append(r.year)
    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        np.asarray(years, dtype=np.int64),
    )


def _rows_to_tensors(rows: list[PanelRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    climates = np.stack([r.climate for r in rows], axis=0).astype(np.float32)
    statics = np.stack([r.static for r in rows], axis=0).astype(np.float32)
    targets = np.array([r.yield_target_pre_biotic_t_ha for r in rows], dtype=np.float32)
    return (
        torch.from_numpy(climates),
        torch.from_numpy(statics),
        torch.from_numpy(targets),
    )


def _quick_train(
    model: QuantileYieldSurrogate,
    train_rows: list[PanelRow],
    *,
    epochs: int = 3,
    device: torch.device,
) -> None:
    from torch.utils.data import DataLoader

    from data.yield_panel import YieldPanelDataset

    loader = DataLoader(YieldPanelDataset(train_rows), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            climate = batch["climate"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(climate, static)
            loss = pinball_loss(pred, target, quantiles=model.quantiles)
            loss.backward()
            opt.step()
    model.eval()


def evaluate_cv_strategy(
    strategy: CvStrategy,
    rows: list[PanelRow],
    *,
    alpha: float = 0.1,
    block_size_km: float = 50.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train a small CQR model and report blocked coverage for one strategy."""
    dev = device or torch.device("cpu")
    lats, lons, years = _panel_coords(rows)
    climate, static, y = _rows_to_tensors(rows)
    n = len(rows)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_cal = max(10, n // 5)
        cal_idx = idx[:n_cal]
        train_idx = idx[n_cal:]
        model = QuantileYieldSurrogate().to(dev)
        train_rows = [rows[i] for i in train_idx]
        _quick_train(model, train_rows, device=dev)
        calibrator = ConformalCalibrator().fit(
            model,
            (climate[cal_idx], static[cal_idx]),
            y[cal_idx],
            alpha=alpha,
            device=dev,
        )
        calibrator.cv_strategy = "random"
        test_idx = idx[n_cal : n_cal + n // 5]
        lowers, _, uppers = calibrator.predict_interval_batch(
            model, (climate[test_idx], static[test_idx]), device=dev
        )
        y_test = y[test_idx].cpu().numpy()
        coverage = float(np.mean((y_test >= lowers) & (y_test <= uppers)))
        pit = _interval_pit(y_test, lowers, uppers).tolist()
        return {
            "strategy": strategy,
            "coverage": coverage,
            "mean_width": float(np.mean(uppers - lowers)),
            "pit": pit,
            "production_target": False,
            "diagnostic_only": True,
        }

    if strategy == "spatial_block":
        splitter = SpatialBlockSplit(
            block_size_km=block_size_km, n_folds=5, strategy="checkerboard", seed=seed
        )
        train_idx = test_idx = None
        for t_idx, te_idx in splitter.split(lats, lons):
            train_idx, test_idx = t_idx, te_idx
            break
        if train_idx is None or test_idx is None:
            raise ValueError("spatial_block split produced no valid fold")
        model = QuantileYieldSurrogate().to(dev)
        _quick_train(model, [rows[i] for i in train_idx], device=dev)
        calibrator = ConformalCalibrator().fit_blocked(
            model,
            (climate, static),
            y,
            splitter,
            coords=(lats, lons),
            alpha=alpha,
            device=dev,
            fold=0,
        )
        lowers, _, uppers = calibrator.predict_interval_batch(
            model, (climate[test_idx], static[test_idx]), device=dev
        )
        y_test = y[test_idx].cpu().numpy()
        coverage = float(np.mean((y_test >= lowers) & (y_test <= uppers)))
        pit = _interval_pit(y_test, lowers, uppers).tolist()
        return {
            "strategy": strategy,
            "coverage": coverage,
            "mean_width": float(np.mean(uppers - lowers)),
            "pit": pit,
            "block_size_km": block_size_km,
            "fold_coverages": calibrator.fold_coverages,
            "production_target": True,
            "diagnostic_only": False,
        }

    if strategy == "buffered_loo":
        splitter = BufferedLOO(buffer_km=block_size_km)
        model = QuantileYieldSurrogate().to(dev)
        _quick_train(model, rows[: max(50, n // 2)], device=dev)
        calibrator = ConformalCalibrator().fit_blocked(
            model,
            (climate, static),
            y,
            splitter,
            coords=(lats, lons),
            alpha=alpha,
            device=dev,
        )
        coverage = float(calibrator.empirical_coverage or 0.0)
        return {
            "strategy": strategy,
            "coverage": coverage,
            "mean_width": float("nan"),
            "pit": [],
            "production_target": False,
            "diagnostic_only": True,
        }

    if strategy == "temporal_forward":
        splitter = ForwardChainSplit(min_train_years=3, max_test_years=1)
        splits = list(splitter.split(years))
        if not splits:
            return {"strategy": strategy, "coverage": float("nan"), "error": "no folds"}
        train_idx, test_idx = splits[-1]
        model = QuantileYieldSurrogate().to(dev)
        _quick_train(model, [rows[i] for i in train_idx], device=dev)
        calibrator = ConformalCalibrator().fit_blocked(
            model,
            (climate, static),
            y,
            splitter,
            years=years,
            alpha=alpha,
            device=dev,
        )
        lowers, _, uppers = calibrator.predict_interval_batch(
            model, (climate[test_idx], static[test_idx]), device=dev
        )
        y_test = y[test_idx].cpu().numpy()
        coverage = float(np.mean((y_test >= lowers) & (y_test <= uppers)))
        pit = _interval_pit(y_test, lowers, uppers).tolist()
        return {
            "strategy": strategy,
            "coverage": coverage,
            "mean_width": float(np.mean(uppers - lowers)),
            "pit": pit,
            "production_target": False,
            "diagnostic_only": True,
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def run_all_cv_strategies(
    *,
    strategies: tuple[CvStrategy, ...] = (
        "random",
        "spatial_block",
        "temporal_forward",
        "buffered_loo",
    ),
    synthetic: bool = True,
    block_size_km: float = 50.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate all strategies on yield panel rows."""
    try:
        rows = build_yield_panel(seed=seed)
    except FileNotFoundError:
        rows = _synthetic_panel_rows(400, seed=seed)
    results: dict[str, Any] = {}
    for name in strategies:
        results[name] = evaluate_cv_strategy(
            name,
            rows,
            block_size_km=block_size_km,
            seed=seed,
        )
    spatial = results.get("spatial_block", {})
    if spatial.get("production_target") and spatial.get("coverage") is not None:
        cov = float(spatial["coverage"])
        if not (COVERAGE_LO <= cov <= COVERAGE_HI):
            results["gate_passed"] = False
            results["gate_message"] = (
                f"spatial_block coverage {cov:.3f} outside [{COVERAGE_LO}, {COVERAGE_HI}]"
            )
        else:
            results["gate_passed"] = True
            results["gate_message"] = "spatial_block coverage within production band"
    return results


def _synthetic_panel_rows(n: int, *, seed: int) -> list[PanelRow]:
    rng = np.random.default_rng(seed)
    out: list[PanelRow] = []
    for i in range(n):
        climate = rng.normal(0, 0.1, (365, 11)).astype(np.float32)
        climate[:, 0] = 28 + rng.normal(0, 1, 365)
        climate[:, 1] = 22 + rng.normal(0, 1, 365)
        climate[:, 2] = 25 + rng.normal(0, 1, 365)
        climate[:, 3] = np.clip(rng.gamma(2, 2, 365), 0, 40)
        climate[:, 4] = 15
        climate[:, 5] = 1.0
        climate[:, 6] = 3.5
        climate[:, 7] = 0.3
        climate[:, 8] = 2.0
        climate[:, 9] = 80
        climate[:, 10] = 415
        static = rng.normal(0, 0.1, 13).astype(np.float32)
        static[0] = 150.0
        y = max(0.2, float(static[0] * 0.005 + climate[:, 2].mean() * 0.04 + rng.normal(0, 0.15)))
        out.append(
            PanelRow(
                sample_id=f"syn_{i}",
                country_iso3="GHA" if i % 2 == 0 else "CIV",
                year=2020 + (i % 5),
                cohort="icco",
                yield_observed_t_ha=y,
                yield_target_pre_biotic_t_ha=y,
                surviving_biotic_fraction=1.0,
                climate=climate,
                static=static,
            )
        )
    return out
