"""BSSAL active learning utilities for sparse-label cocoa expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier

EPS = 1e-12


@dataclass(frozen=True)
class ActiveLearningBatch:
    """Ranked candidate query set after entropy and spatial filtering."""

    indices: np.ndarray
    entropy: np.ndarray
    probabilities: np.ndarray
    range_m: float


def vote_entropy(votes: np.ndarray, *, n_classes: int | None = None) -> np.ndarray:
    """Compute normalized vote entropy from committee class votes."""
    vote_arr = np.asarray(votes)
    if vote_arr.ndim != 2:
        raise ValueError(f"votes must be [n_committee, n_samples], got {vote_arr.shape}")
    classes = int(n_classes or (np.max(vote_arr) + 1 if vote_arr.size else 2))
    counts = np.zeros((vote_arr.shape[1], classes), dtype=np.float64)
    for member_votes in vote_arr:
        for class_id in range(classes):
            counts[:, class_id] += member_votes == class_id
    probs = counts / max(vote_arr.shape[0], 1)
    safe_log = np.zeros_like(probs, dtype=np.float64)
    np.log(probs, out=safe_log, where=probs > 0)
    entropy = -(probs * safe_log).sum(axis=1)
    return entropy / max(np.log(classes), EPS)


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Return a WGS84 UTM EPSG code for the coordinate."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32700 if lat < 0 else 32600) + zone


def project_lonlat_to_utm(coords: np.ndarray, *, epsg: int | None = None) -> np.ndarray:
    """Project ``[lon, lat]`` coordinates to UTM meters."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must be [n, 2] lon/lat, got {arr.shape}")
    if len(arr) == 0:
        return arr.copy()
    target_epsg = epsg or utm_epsg_for_lonlat(float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    x, y = transformer.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([x, y]).astype(np.float64)


def spatial_uncorrelation_mask(
    candidate_lonlat: np.ndarray,
    labeled_lonlat: np.ndarray,
    range_m: float,
) -> np.ndarray:
    """Keep candidates at least ``range_m`` from every labeled sample."""
    candidates = np.asarray(candidate_lonlat, dtype=np.float64)
    labeled = np.asarray(labeled_lonlat, dtype=np.float64)
    if len(candidates) == 0:
        return np.zeros(0, dtype=bool)
    if len(labeled) == 0 or range_m <= 0:
        return np.ones(len(candidates), dtype=bool)
    epsg = utm_epsg_for_lonlat(
        float(np.concatenate([candidates[:, 0], labeled[:, 0]]).mean()),
        float(np.concatenate([candidates[:, 1], labeled[:, 1]]).mean()),
    )
    cand_xy = project_lonlat_to_utm(candidates, epsg=epsg)
    label_xy = project_lonlat_to_utm(labeled, epsg=epsg)
    keep = np.ones(len(candidates), dtype=bool)
    chunk = 2048
    for start in range(0, len(candidates), chunk):
        end = start + chunk
        distances = np.linalg.norm(cand_xy[start:end, None, :] - label_xy[None, :, :], axis=2)
        keep[start:end] = distances.min(axis=1) >= range_m
    return keep


class BSSALCocoaLearner:
    """Two-member RF query-by-committee learner with variogram spatial filtering."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        random_state: int = 42,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
    ) -> None:
        self.random_state = random_state
        self.committee = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=random_state + 1,
                n_jobs=-1,
            ),
        ]
        self.classes_: np.ndarray | None = None
        self.spatial_range_m_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> BSSALCocoaLearner:
        """Fit both RF committee members on labeled cocoa exposure samples."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be [n_samples, n_features], got {X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have the same number of rows")
        for member in self.committee:
            member.fit(X_arr, y_arr)
        self.classes_ = self.committee[0].classes_
        return self

    def committee_votes(self, X: np.ndarray) -> np.ndarray:
        """Return class votes as ``[2, n_samples]``."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.vstack([member.predict(X_arr) for member in self.committee])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average committee class probabilities."""
        X_arr = np.asarray(X, dtype=np.float64)
        probs = [member.predict_proba(X_arr) for member in self.committee]
        return np.mean(probs, axis=0)

    def vote_entropy(self, X: np.ndarray) -> np.ndarray:
        """Rank uncertainty from two-member QBC vote entropy."""
        votes = self.committee_votes(X)
        n_classes = len(self.classes_) if self.classes_ is not None else None
        return vote_entropy(votes, n_classes=n_classes)

    def fit_monthly_variogram_ranges(
        self,
        lonlat: np.ndarray,
        monthly_ndvi: np.ndarray,
        *,
        maxlag: str | float | None = "median",
        n_lags: int = 12,
    ) -> np.ndarray:
        """Fit 12 monthly NDVI semi-variograms and store the minimum range in meters."""
        coords = np.asarray(lonlat, dtype=np.float64)
        ndvi = np.asarray(monthly_ndvi, dtype=np.float64)
        if ndvi.ndim != 2 or ndvi.shape[1] != 12:
            raise ValueError(f"monthly_ndvi must be [n_samples, 12], got {ndvi.shape}")
        if len(coords) != len(ndvi):
            raise ValueError("lonlat and monthly_ndvi must have the same number of rows")
        xy = project_lonlat_to_utm(coords)
        ranges: list[float] = []
        try:
            from skgstat import Variogram
        except ImportError:
            Variogram = None
        for month in range(12):
            values = ndvi[:, month]
            valid = np.isfinite(values)
            if valid.sum() < 6:
                continue
            if Variogram is None:
                ranges.append(_fallback_variogram_range(xy[valid], values[valid]))
                continue
            variogram = Variogram(
                xy[valid],
                values[valid],
                model="spherical",
                n_lags=n_lags,
                maxlag=maxlag,
                normalize=False,
            )
            fitted_range = float(variogram.parameters[0])
            if np.isfinite(fitted_range) and fitted_range > 0:
                ranges.append(fitted_range)
        if not ranges:
            raise ValueError("Unable to fit any monthly NDVI variogram ranges")
        out = np.asarray(ranges, dtype=np.float64)
        self.spatial_range_m_ = float(np.min(out))
        return out

    def query(
        self,
        candidate_X: np.ndarray,
        candidate_lonlat: np.ndarray,
        labeled_lonlat: np.ndarray,
        *,
        budget: int,
        monthly_ndvi: np.ndarray | None = None,
        range_m: float | None = None,
    ) -> ActiveLearningBatch:
        """Query top-entropy candidates after semi-variogram spatial filtering."""
        if budget <= 0:
            raise ValueError("budget must be positive")
        candidate_arr = np.asarray(candidate_X, dtype=np.float64)
        if range_m is None:
            if self.spatial_range_m_ is None:
                if monthly_ndvi is None:
                    raise ValueError("monthly_ndvi or range_m is required before querying")
                self.fit_monthly_variogram_ranges(candidate_lonlat, monthly_ndvi)
            range_m = float(self.spatial_range_m_)
        keep = spatial_uncorrelation_mask(candidate_lonlat, labeled_lonlat, float(range_m))
        entropy = self.vote_entropy(candidate_arr)
        probabilities = self.predict_proba(candidate_arr)
        eligible = np.flatnonzero(keep)
        ranked = eligible[np.argsort(-entropy[eligible], kind="mergesort")]
        selected = ranked[:budget]
        return ActiveLearningBatch(
            indices=selected,
            entropy=entropy[selected],
            probabilities=probabilities[selected],
            range_m=float(range_m),
        )


def _fallback_variogram_range(xy: np.ndarray, values: np.ndarray) -> float:
    """Empirical range fallback used only when scikit-gstat is unavailable."""
    centered = values - float(np.mean(values))
    sill = float(np.var(centered))
    if sill <= 0:
        return 0.0
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    semivar = 0.5 * (centered[:, None] - centered[None, :]) ** 2
    bins = np.linspace(0.0, float(np.max(distances)), 13)[1:]
    for edge in bins:
        mask = (distances > 0) & (distances <= edge)
        if np.any(mask) and float(np.mean(semivar[mask])) >= 0.95 * sill:
            return float(edge)
    return float(np.median(distances[distances > 0]))


class BayesianHead(nn.Module):
    """Dropout BNN head for frozen ``ensemble_v2`` feature backbones."""

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        in_features: int = 256,
        hidden_features: int = 128,
        out_features: int = 1,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if self.backbone is not None and freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def _features(self, x: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if self.backbone is None:
            if not torch.is_tensor(x):
                raise TypeError("BayesianHead without a backbone expects a tensor")
            features = x
        else:
            with torch.no_grad():
                features = self.backbone(x)
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

    def forward(self, x: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        return self.head(self._features(x))

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor | dict[str, Any],
        *,
        mc_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MC-dropout mean and standard deviation probabilities."""
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive")
        was_training = self.training
        self.train()
        outputs = []
        for _ in range(mc_samples):
            outputs.append(torch.sigmoid(self.forward(x)))
        if not was_training:
            self.eval()
        stacked = torch.stack(outputs, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)


def load_bayesian_head_checkpoint(
    path: Path | str,
    *,
    backbone: nn.Module | None = None,
    device: str = "cpu",
) -> BayesianHead:
    """Load a BSSAL Bayesian head checkpoint with CI-safe defaults."""
    state = torch.load(Path(path), map_location=device, weights_only=False)
    kwargs = state.get("model_kwargs", {}) if isinstance(state, dict) else {}
    model = BayesianHead(backbone=backbone, **kwargs).to(device)
    weights = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


__all__ = [
    "ActiveLearningBatch",
    "BSSALCocoaLearner",
    "BayesianHead",
    "load_bayesian_head_checkpoint",
    "project_lonlat_to_utm",
    "spatial_uncorrelation_mask",
    "utm_epsg_for_lonlat",
    "vote_entropy",
]
