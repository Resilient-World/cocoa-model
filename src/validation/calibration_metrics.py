"""CQR / conformal calibration metrics for CI gates and MODEL_CARD."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.yield_panel import PanelRow
from models.conformal.cqr import DEFAULT_QUANTILES, ConformalCalibrator, QuantileYieldSurrogate
from validation.baselines import evaluate_with_baselines, group_indices_by_stratum, panel_stratum
from validation.conformal_cv import _panel_coords, _quick_train, _rows_to_tensors
from validation.forecast_scoring import (
    crps_ensemble,
    crps_quantile,
    pit_histogram,
    reliability_diagram,
    sharpness,
)
from validation.spatial_cv import SpatialBlockSplit

NOMINAL_COVERAGE = 0.90
COVERAGE_TOL = 0.02
PIT_CHI2_MIN_P = 0.01
SHARPNESS_REGRESSION_MAX = 1.10


@dataclass
class CalibrationReport:
    model: str
    nominal_coverage: float
    empirical_coverage: float
    crps: float
    ece: float
    sharpness: float
    pit_chi2_p: float
    pit_shape: str
    crpss_climatology: float
    crpss_persistence: float
    crpss_fdp_mean: float
    by_stratum: dict[str, dict[str, Any]] = field(default_factory=dict)
    pit: list[float] = field(default_factory=list)
    reliability_nominal: list[float] = field(default_factory=list)
    reliability_empirical: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics_block(
    y: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    q_preds: np.ndarray,
    model_ens: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    nominal = 1.0 - alpha
    coverage = float(np.mean((y >= lowers) & (y <= uppers)))
    _, _, rel_data, pit_diag = pit_histogram(y, lowers=lowers, uppers=uppers)
    nom, emp, ece, _ = reliability_diagram(y, q_preds, DEFAULT_QUANTILES)
    width = sharpness((lowers, uppers))
    assert isinstance(width, float)
    crps_q = float(np.nanmean(crps_quantile(y, q_preds, DEFAULT_QUANTILES)))
    crps_e = float(np.nanmean(crps_ensemble(y, model_ens)))
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": coverage,
        "crps": crps_e,
        "crps_quantile": crps_q,
        "ece": ece,
        "sharpness": width,
        "pit_chi2_p": pit_diag["pit_chi2_p"],
        "pit_shape": pit_diag["shape"],
        "pit": pit_histogram(y, lowers=lowers, uppers=uppers)[2],
        "reliability_nominal": nom.tolist(),
        "reliability_empirical": emp.tolist(),
        "pit_counts": pit_histogram(y, lowers=lowers, uppers=uppers)[1].tolist(),
    }


def evaluate_cqr_calibration(
    rows: list[PanelRow],
    *,
    alpha: float = 0.1,
    block_size_km: float = 50.0,
    seed: int = 42,
    device: torch.device | None = None,
    scenario: str = "ssp245",
) -> CalibrationReport:
    """Spatial-block CQR evaluation with pooled and per-stratum metrics."""
    dev = device or torch.device("cpu")
    lats, lons, _years = _panel_coords(rows)
    climate, static, y_t = _rows_to_tensors(rows)
    y = y_t.cpu().numpy()
    n = len(rows)

    splitter = SpatialBlockSplit(
        block_size_km=block_size_km, n_folds=5, strategy="checkerboard", seed=seed
    )
    train_idx = test_idx = None
    for t_idx, te_idx in splitter.split(lats, lons):
        train_idx, test_idx = t_idx, te_idx
        break
    if train_idx is None or test_idx is None:
        raise ValueError("spatial_block produced no valid fold")

    model = QuantileYieldSurrogate().to(dev)
    _quick_train(model, [rows[i] for i in train_idx], device=dev)
    calibrator = ConformalCalibrator().fit_blocked(
        model,
        (climate, static),
        y_t,
        splitter,
        coords=(lats, lons),
        alpha=alpha,
        device=dev,
        fold=0,
    )

    with torch.no_grad():
        q_raw = model(climate[test_idx].to(dev), static[test_idx].to(dev)).cpu().numpy()
    lowers, _, uppers = calibrator.predict_interval_batch(
        model, (climate[test_idx], static[test_idx]), device=dev
    )
    y_test = y[test_idx]
    spread = (q_raw[:, 2] - q_raw[:, 0]).reshape(-1, 1)
    model_ens = q_raw[:, 1:2] + np.linspace(-1, 1, 11) * spread

    pooled = _metrics_block(y_test, lowers, uppers, q_raw, model_ens, alpha=alpha)
    baseline_stats = evaluate_with_baselines(
        y_test, model_ens, rows, test_idx, scenario=scenario, seed=seed
    )

    by_stratum: dict[str, dict[str, Any]] = {}
    pos_map = {int(test_idx[k]): k for k in range(len(test_idx))}
    for key, idx in group_indices_by_stratum(rows, test_idx, scenario=scenario).items():
        local = np.array([pos_map[int(i)] for i in idx if int(i) in pos_map], dtype=np.int64)
        if local.size == 0:
            continue
        sub = _metrics_block(
            y_test[local],
            lowers[local],
            uppers[local],
            q_raw[local],
            model_ens[local],
            alpha=alpha,
        )
        sub["stratum_key"] = key
        by_stratum[key] = sub

    pit_vals = (
        (y_test - lowers) / np.maximum(uppers - lowers, 1e-6)
    ).clip(0, 1).tolist()

    return CalibrationReport(
        model="cqr_yield",
        nominal_coverage=float(pooled["nominal_coverage"]),
        empirical_coverage=float(pooled["empirical_coverage"]),
        crps=float(pooled["crps"]),
        ece=float(pooled["ece"]),
        sharpness=float(pooled["sharpness"]),
        pit_chi2_p=float(pooled["pit_chi2_p"]),
        pit_shape=str(pooled["pit_shape"]),
        crpss_climatology=float(baseline_stats["crpss_climatology"]),
        crpss_persistence=float(baseline_stats["crpss_persistence"]),
        crpss_fdp_mean=float(baseline_stats["crpss_fdp_mean"]),
        by_stratum=by_stratum,
        pit=pit_vals,
        reliability_nominal=pooled["reliability_nominal"],
        reliability_empirical=pooled["reliability_empirical"],
    )


def run_calibration_gate(
    report: CalibrationReport | dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    enforce_sharpness: bool = True,
) -> tuple[bool, list[str]]:
    """Return (passed, messages)."""
    if isinstance(report, CalibrationReport):
        data = report.to_dict()
    else:
        data = report
    messages: list[str] = []
    ok = True
    nom = float(data.get("nominal_coverage", NOMINAL_COVERAGE))
    emp = float(data["empirical_coverage"])
    if abs(emp - nom) > COVERAGE_TOL:
        ok = False
        messages.append(f"coverage |{emp:.3f} - {nom:.3f}| > {COVERAGE_TOL}")
    pit_p = float(data.get("pit_chi2_p", 1.0))
    if pit_p < PIT_CHI2_MIN_P:
        ok = False
        messages.append(f"PIT chi2 p={pit_p:.4f} < {PIT_CHI2_MIN_P} (shape={data.get('pit_shape')})")
    if enforce_sharpness and baseline is not None:
        base_w = float(baseline.get("sharpness", float("nan")))
        cur_w = float(data["sharpness"])
        if np.isfinite(base_w) and base_w > 0 and cur_w > base_w * SHARPNESS_REGRESSION_MAX:
            ok = False
            messages.append(
                f"sharpness {cur_w:.4f} > {SHARPNESS_REGRESSION_MAX:.0%} × baseline {base_w:.4f}"
            )
    if ok:
        messages.append("calibration gate passed")
    return ok, messages


def write_calibration_report(
    report: CalibrationReport,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON + markdown under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    json_path = out_dir / f"calibration_{report.model}_{day}.json"
    md_path = out_dir / f"calibration_{report.model}_{day}.md"
    payload = report.to_dict()
    payload["date"] = day
    import json

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "calibration_latest.json"
    latest.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        f"# Calibration report — {report.model}",
        "",
        f"Date: {day}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| CRPS | {report.crps:.4f} |",
        f"| CRPSS (climatology) | {report.crpss_climatology:.4f} |",
        f"| CRPSS (persistence) | {report.crpss_persistence:.4f} |",
        f"| CRPSS (FDP mean) | {report.crpss_fdp_mean:.4f} |",
        f"| ECE | {report.ece:.4f} |",
        f"| PIT χ² p | {report.pit_chi2_p:.4f} |",
        f"| PIT shape | {report.pit_shape} |",
        f"| Sharpness | {report.sharpness:.4f} |",
        f"| Coverage | {report.empirical_coverage:.3f} (nominal {report.nominal_coverage:.2f}) |",
        "",
        "## Per-stratum",
        "",
        "| Stratum | CRPS | ECE | Coverage |",
        "|---------|------|-----|----------|",
    ]
    for sk, sub in report.by_stratum.items():
        lines.append(
            f"| {sk} | {sub.get('crps', float('nan')):.4f} | "
            f"{sub.get('ece', float('nan')):.4f} | "
            f"{sub.get('empirical_coverage', float('nan')):.3f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "CalibrationReport",
    "COVERAGE_TOL",
    "PIT_CHI2_MIN_P",
    "evaluate_cqr_calibration",
    "run_calibration_gate",
    "write_calibration_report",
]
