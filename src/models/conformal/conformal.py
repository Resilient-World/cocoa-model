"""
Split and Mondrian conformal prediction for cocoa yield (tonnes/ha).

Nonconformity scores normalize absolute residuals by MC-dropout uncertainty::

    s = |y - ŷ| / (σ_MC + ε)

Intervals are ``ŷ ± q̂ · (σ_MC + ε)`` where ``q̂`` is a calibrated quantile of ``s``.
Under exchangeability, split conformal prediction attains finite-sample coverage
≥ 1 − α on the calibration distribution; Mondrian stratification applies separate
quantiles per Kalischek (2023) West African cocoa agroecological zone.
"""

from __future__ import annotations

import structlog

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from models.surrogate.yield_surrogate import YieldSurrogateModel, predict_with_uncertainty

log = structlog.get_logger(__name__)

DEFAULT_EPSILON = 1e-6
DEFAULT_NUM_MC_SAMPLES = 50

# Kalischek et al. (2023) West African cocoa belt agroecological zones
KALISCHEK_ZONES: tuple[str, ...] = (
    "Forest",
    "Forest-Savanna Transition",
    "Guinea Savanna",
)

COVERAGE_GUARANTEE_SPLIT = (
    "Split conformal prediction: empirical coverage on exchangeable test points "
    "is at least {coverage:.0%} in expectation (finite-sample guarantee ≥ 1−α on "
    "the calibration split when scores are exchangeable)."
)

COVERAGE_GUARANTEE_MONDRIAN = (
    "Mondrian (regional) conformal prediction: per-zone quantiles give marginal "
    "coverage ≥ {coverage:.0%} within each calibrated agroecological zone under "
    "exchangeability within strata."
)


@dataclass
class ConformalInterval:
    """Prediction interval with explicit coverage target and method metadata."""

    point: float
    lower: float
    upper: float
    coverage_target: float
    method: str = "split_conformal"
    coverage_guarantee: str = ""

    def __post_init__(self) -> None:
        if not self.coverage_guarantee:
            self.coverage_guarantee = COVERAGE_GUARANTEE_SPLIT.format(
                coverage=self.coverage_target
            )


@dataclass
class _CalibrationBatch:
    climate: Tensor
    static: Tensor
    y: Tensor
    zone: list[str] | None = None


def nonconformity_score(
    y: Tensor,
    y_hat: Tensor,
    sigma_mc: Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> Tensor:
    """|y − ŷ| / (σ_MC + ε), shape broadcast-compatible with batch dimension."""
    return (y - y_hat).abs() / (sigma_mc + epsilon)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal critical value (1−α) from calibration scores.

    Uses the standard finite-sample rank ``k = ceil((n+1)(1−α))``.
    """
    if scores.size == 0:
        raise ValueError("Cannot compute conformal quantile from empty scores")
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def assign_kalischek_zone(lat: float, lon: float) -> str:
    """
  Assign a Kalischek (2023) West African cocoa agroecological zone from coordinates.

    Heuristic latitudinal bands within the cocoa belt (~4–11°N, 12°W–5°E).
    """
    if not (-12.0 <= lon <= 5.0 and 4.0 <= lat <= 11.0):
        return "Forest-Savanna Transition"
    if lat < 6.5:
        return "Forest"
    if lat < 8.5:
        return "Forest-Savanna Transition"
    return "Guinea Savanna"


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations with ``y`` in ``[lower, upper]``."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


class CalibrationParquetDataset(Dataset):
    """
    Calibration rows from parquet: ``lat``, ``lon``, ``yield_tpha``, optional ``year``.

    Climate/static tensors are built via ``feature_builder(lat, lon, year)`` when
    arrays are not stored in the file.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_builder: Any | None = None,
        *,
        year_col: str = "year",
        y_col: str = "yield_tpha",
    ) -> None:
        required = {"lat", "lon", y_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Calibration parquet missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self.feature_builder = feature_builder
        self.year_col = year_col
        self.y_col = y_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        lat = float(row["lat"])
        lon = float(row["lon"])
        year = int(row[self.year_col]) if self.year_col in self.df.columns else 2023
        y = float(row[self.y_col])

        if self.feature_builder is not None:
            climate, static = self.feature_builder(lat, lon, year)
        else:
            for col in ("climate", "static"):
                if col not in self.df.columns:
                    raise ValueError(
                        "Provide feature_builder or precomputed 'climate'/'static' columns"
                    )
            climate = torch.as_tensor(row["climate"], dtype=torch.float32)
            static = torch.as_tensor(row["static"], dtype=torch.float32)

        zone = row["agroecological_zone"] if "agroecological_zone" in self.df.columns else assign_kalischek_zone(lat, lon)

        return {
            "climate": climate,
            "static": static,
            "y": torch.tensor(y, dtype=torch.float32),
            "zone": str(zone),
        }


def _collate_calibration(batch: list[dict[str, Any]]) -> _CalibrationBatch:
    climate = torch.stack([b["climate"] for b in batch], dim=0)
    static = torch.stack([b["static"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    zones = [b["zone"] for b in batch]
    return _CalibrationBatch(climate=climate, static=static, y=y, zone=zones)


def _iter_calibration_scores(
    model: YieldSurrogateModel,
    calib_loader: DataLoader,
    *,
    num_samples: int,
    epsilon: float,
    device: str,
) -> tuple[np.ndarray, list[str] | None]:
    """Collect nonconformity scores (and optional zones) from a calibration loader."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_zones: list[str] = []

    with torch.no_grad():
        for batch in calib_loader:
            if isinstance(batch, _CalibrationBatch):
                cb = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                climate_b, static_b, y_b = batch[0], batch[1], batch[2]
                cb = _CalibrationBatch(
                    climate=climate_b,
                    static=static_b,
                    y=y_b,
                    zone=None,
                )
            elif isinstance(batch, list) and batch and isinstance(batch[0], dict):
                cb = _collate_calibration(batch)
            else:
                raise TypeError(f"Unsupported calibration batch type: {type(batch)}")

            climate = cb.climate.to(device)
            static = cb.static.to(device)
            y = cb.y.to(device)

            pred = predict_with_uncertainty(model, climate, static, num_samples=num_samples)
            scores = nonconformity_score(y, pred.mean, pred.std, epsilon=epsilon)
            all_scores.append(scores.cpu().numpy())

            if cb.zone is not None:
                all_zones.extend(cb.zone)

    scores_arr = np.concatenate(all_scores, axis=0)
    zones_out = all_zones if all_zones else None
    return scores_arr, zones_out


@dataclass
class SplitConformalYield:
    """
    Split conformal predictor wrapping :func:`predict_with_uncertainty`.
    """

    quantile: float = 0.0
    alpha: float = 0.1
    epsilon: float = DEFAULT_EPSILON
    num_mc_samples: int = DEFAULT_NUM_MC_SAMPLES
    coverage_target: float = field(init=False)
    validation: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.coverage_target = 1.0 - self.alpha

    def calibrate(
        self,
        model: YieldSurrogateModel,
        calib_loader: DataLoader,
        alpha: float = 0.1,
        *,
        num_samples: int | None = None,
        epsilon: float | None = None,
        device: str = "cpu",
    ) -> SplitConformalYield:
        """Fit critical value ``q̂`` from calibration nonconformity scores."""
        self.alpha = alpha
        self.coverage_target = 1.0 - alpha
        mc = num_samples if num_samples is not None else self.num_mc_samples
        eps = epsilon if epsilon is not None else self.epsilon

        scores, _ = _iter_calibration_scores(
            model,
            calib_loader,
            num_samples=mc,
            epsilon=eps,
            device=device,
        )
        n = scores.size
        split = max(1, int(0.8 * n))
        train_scores = scores[:split]
        holdout_scores = scores[split:] if split < n else scores
        self.quantile = conformal_quantile(train_scores, alpha)
        empirical = float(np.mean(holdout_scores <= self.quantile)) if holdout_scores.size else 1.0
        self.validation = {
            "empirical_coverage": empirical,
            "nominal_coverage": self.coverage_target,
            "n_holdout": float(holdout_scores.size),
        }
        self.num_mc_samples = mc
        self.epsilon = eps
        log.info(
            "Split conformal calibrated: n=%d, q_hat=%.4f, holdout coverage=%.3f",
            scores.size,
            self.quantile,
            empirical,
        )
        return self

    @torch.no_grad()
    def predict(
        self,
        model: YieldSurrogateModel,
        climate: Tensor,
        static: Tensor,
        *,
        num_samples: int | None = None,
        device: str = "cpu",
    ) -> ConformalInterval:
        """Return conformal interval for a single or batched site (batch size 1 typical)."""
        if self.quantile <= 0.0:
            raise RuntimeError("Call calibrate() before predict()")

        mc = num_samples if num_samples is not None else self.num_mc_samples
        climate = climate.to(device)
        static = static.to(device)

        pred = predict_with_uncertainty(model, climate, static, num_samples=mc)
        point = float(pred.mean.squeeze().item())
        sigma = float(pred.std.squeeze().item())
        half_width = self.quantile * (sigma + self.epsilon)

        return ConformalInterval(
            point=point,
            lower=point - half_width,
            upper=point + half_width,
            coverage_target=self.coverage_target,
            method="split_conformal",
            coverage_guarantee=COVERAGE_GUARANTEE_SPLIT.format(coverage=self.coverage_target),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": "split",
            "quantile": self.quantile,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "num_mc_samples": self.num_mc_samples,
            "coverage_target": self.coverage_target,
        }
        if self.validation is not None:
            payload["validation"] = self.validation
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitConformalYield:
        obj = cls(
            quantile=float(data["quantile"]),
            alpha=float(data.get("alpha", 0.1)),
            epsilon=float(data.get("epsilon", DEFAULT_EPSILON)),
            num_mc_samples=int(data.get("num_mc_samples", DEFAULT_NUM_MC_SAMPLES)),
        )
        obj.coverage_target = float(data.get("coverage_target", 1.0 - obj.alpha))
        obj.validation = data.get("validation")
        return obj


@dataclass
class MondrianConformalYield:
    """
    Mondrian conformal prediction with per–agro-ecological-zone quantiles.
    """

    zone_quantiles: dict[str, float] = field(default_factory=dict)
    fallback_quantile: float = 0.0
    alpha: float = 0.1
    epsilon: float = DEFAULT_EPSILON
    num_mc_samples: int = DEFAULT_NUM_MC_SAMPLES
    coverage_target: float = field(init=False)
    validation: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.coverage_target = 1.0 - self.alpha

    def calibrate(
        self,
        model: YieldSurrogateModel,
        calib_loader: DataLoader,
        alpha: float = 0.1,
        *,
        num_samples: int | None = None,
        epsilon: float | None = None,
        device: str = "cpu",
        min_zone_samples: int = 30,
    ) -> MondrianConformalYield:
        """Fit per-zone quantiles; global fallback for sparse zones."""
        self.alpha = alpha
        self.coverage_target = 1.0 - alpha
        mc = num_samples if num_samples is not None else self.num_mc_samples
        eps = epsilon if epsilon is not None else self.epsilon

        scores, zones = _iter_calibration_scores(
            model,
            calib_loader,
            num_samples=mc,
            epsilon=eps,
            device=device,
        )
        if zones is None:
            raise ValueError("Mondrian calibration requires zone labels on each batch")

        self.fallback_quantile = conformal_quantile(scores, alpha)
        self.zone_quantiles = {}

        scores_by_zone: dict[str, list[float]] = {z: [] for z in KALISCHEK_ZONES}
        for score, zone in zip(scores.tolist(), zones, strict=True):
            bucket = scores_by_zone.setdefault(zone, [])
            bucket.append(score)

        for zone, zone_scores in scores_by_zone.items():
            if len(zone_scores) >= min_zone_samples:
                self.zone_quantiles[zone] = conformal_quantile(
                    np.asarray(zone_scores, dtype=np.float64),
                    alpha,
                )
            else:
                log.warning(
                    "Zone %s has only %d calibration points; using global fallback",
                    zone,
                    len(zone_scores),
                )

        split = max(1, int(0.8 * scores.size))
        holdout_scores = scores[split:] if split < scores.size else scores
        q_global = self.fallback_quantile
        empirical = float(np.mean(holdout_scores <= q_global)) if holdout_scores.size else 1.0
        self.validation = {
            "empirical_coverage": empirical,
            "nominal_coverage": self.coverage_target,
            "n_holdout": float(holdout_scores.size),
        }
        self.num_mc_samples = mc
        self.epsilon = eps
        return self

    def _quantile_for_zone(self, zone: str) -> float:
        return self.zone_quantiles.get(zone, self.fallback_quantile)

    @torch.no_grad()
    def predict(
        self,
        model: YieldSurrogateModel,
        climate: Tensor,
        static: Tensor,
        *,
        lat: float | None = None,
        lon: float | None = None,
        zone: str | None = None,
        num_samples: int | None = None,
        device: str = "cpu",
    ) -> ConformalInterval:
        if not self.zone_quantiles and self.fallback_quantile <= 0.0:
            raise RuntimeError("Call calibrate() before predict()")

        resolved_zone = zone
        if resolved_zone is None:
            if lat is None or lon is None:
                raise ValueError("Provide zone or (lat, lon) for Mondrian prediction")
            resolved_zone = assign_kalischek_zone(lat, lon)

        q_hat = self._quantile_for_zone(resolved_zone)
        mc = num_samples if num_samples is not None else self.num_mc_samples
        climate = climate.to(device)
        static = static.to(device)

        pred = predict_with_uncertainty(model, climate, static, num_samples=mc)
        point = float(pred.mean.squeeze().item())
        sigma = float(pred.std.squeeze().item())
        half_width = q_hat * (sigma + self.epsilon)

        return ConformalInterval(
            point=point,
            lower=point - half_width,
            upper=point + half_width,
            coverage_target=self.coverage_target,
            method=f"mondrian_conformal:{resolved_zone}",
            coverage_guarantee=COVERAGE_GUARANTEE_MONDRIAN.format(
                coverage=self.coverage_target
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": "mondrian",
            "zone_quantiles": self.zone_quantiles,
            "fallback_quantile": self.fallback_quantile,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "num_mc_samples": self.num_mc_samples,
            "coverage_target": self.coverage_target,
        }
        if self.validation is not None:
            payload["validation"] = self.validation
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MondrianConformalYield:
        obj = cls(
            zone_quantiles=dict(data.get("zone_quantiles", {})),
            fallback_quantile=float(data.get("fallback_quantile", 0.0)),
            alpha=float(data.get("alpha", 0.1)),
            epsilon=float(data.get("epsilon", DEFAULT_EPSILON)),
            num_mc_samples=int(data.get("num_mc_samples", DEFAULT_NUM_MC_SAMPLES)),
        )
        obj.coverage_target = float(data.get("coverage_target", 1.0 - obj.alpha))
        obj.validation = data.get("validation")
        return obj


ConformalPredictor = SplitConformalYield | MondrianConformalYield


def save_conformal(predictor: ConformalPredictor, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(predictor.to_dict(), handle, indent=2)


def load_conformal(path: Path | str) -> ConformalPredictor:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    kind = data.get("kind", "split")
    if kind == "mondrian":
        return MondrianConformalYield.from_dict(data)
    return SplitConformalYield.from_dict(data)


def load_conformal_if_exists(path: Path | str | None) -> ConformalPredictor | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return load_conformal(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_feature_builder() -> Any:
    from api.feature_resolver import FarmFeatureResolver, FeatureResolverConfig

    resolver = FarmFeatureResolver(FeatureResolverConfig())

    def builder(lat: float, lon: float, year: int) -> tuple[Tensor, Tensor]:
        climate = resolver.resolve_climate(lat, lon, year).squeeze(0)
        static = resolver.resolve_static(lat, lon).squeeze(0)
        return climate, static

    return builder


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from api.model_loader import load_yield_model

    checkpoint = Path(args.checkpoint)
    calib_path = Path(args.calib)
    out_path = Path(args.out)

    if not calib_path.is_file():
        log.error("Calibration file not found: %s", calib_path)
        return 1

    df = pd.read_parquet(calib_path)
    feature_builder = None if args.precomputed_features else _build_feature_builder()
    dataset = CalibrationParquetDataset(df, feature_builder)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_calibration,
    )

    model = load_yield_model(str(checkpoint) if checkpoint.is_file() else None)
    alpha = args.alpha

    if args.mondrian:
        predictor: ConformalPredictor = MondrianConformalYield().calibrate(
            model,
            loader,
            alpha=alpha,
            num_samples=args.num_samples,
            device=args.device,
        )
    else:
        predictor = SplitConformalYield().calibrate(
            model,
            loader,
            alpha=alpha,
            num_samples=args.num_samples,
            device=args.device,
        )

    save_conformal(predictor, out_path)
    log.info("Wrote conformal calibration to %s", out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conformal calibration for YieldSurrogateModel",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Fit conformal quantiles from calibration parquet")
    cal.add_argument("--checkpoint", type=Path, default=Path("models/yield.pt"))
    cal.add_argument("--calib", type=Path, default=Path("data/processed/calib.parquet"))
    cal.add_argument("--out", type=Path, default=Path("models/conformal.json"))
    cal.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (default 0.1 → 90%% CI)")
    cal.add_argument("--batch-size", type=int, default=32)
    cal.add_argument("--num-samples", type=int, default=DEFAULT_NUM_MC_SAMPLES)
    cal.add_argument("--device", type=str, default="cpu")
    cal.add_argument(
        "--mondrian",
        action="store_true",
        help="Mondrian stratification by Kalischek agroecological zone",
    )
    cal.add_argument(
        "--precomputed-features",
        action="store_true",
        help="Parquet stores climate/static arrays (skip feature resolver)",
    )
    cal.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
