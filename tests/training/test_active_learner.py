from __future__ import annotations

import numpy as np
import pytest
import torch

from training.active_learner import (
    BSSALCocoaLearner,
    spatial_uncorrelation_mask,
    vote_entropy,
)
from training.ssl_pseudo import make_pseudo_labels


def _gaussian_mixture(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    neg = rng.normal(loc=(-1.0, -1.0), scale=0.35, size=(80, 2))
    pos = rng.normal(loc=(1.0, 1.0), scale=0.35, size=(80, 2))
    X = np.vstack([neg, pos])
    y = np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
    lonlat = np.column_stack([-75.0 + X[:, 0] * 0.01, -8.0 + X[:, 1] * 0.01])
    month = np.arange(12)
    ndvi = 0.5 + 0.1 * np.sin(month[None, :] / 12 * 2 * np.pi) + 0.03 * X[:, :1]
    return X, y, lonlat, ndvi


def test_spatial_filtering_reduces_candidate_count() -> None:
    _, _, lonlat, _ = _gaussian_mixture()
    candidates = lonlat[:20]
    labeled = lonlat[:5]
    keep = spatial_uncorrelation_mask(candidates, labeled, range_m=500)
    assert keep.sum() < len(candidates)
    assert keep.sum() > 0


def test_vote_entropy_ranking_is_correct() -> None:
    votes = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ]
    )
    entropy = vote_entropy(votes, n_classes=2)
    ranked = np.argsort(-entropy, kind="mergesort")
    assert ranked[:2].tolist() == [1, 2]
    assert entropy[0] == pytest.approx(0.0)
    assert entropy[3] == pytest.approx(0.0)
    assert entropy[1] == pytest.approx(1.0)


def test_bssal_queries_highest_entropy_after_spatial_filter() -> None:
    X, y, lonlat, ndvi = _gaussian_mixture()
    learner = BSSALCocoaLearner(n_estimators=30, random_state=3).fit(X, y)
    query = learner.query(
        X,
        lonlat,
        lonlat[:8],
        budget=10,
        monthly_ndvi=ndvi,
        range_m=2_000,
    )
    assert len(query.indices) <= 10
    assert query.range_m == pytest.approx(2_000)
    assert np.all(query.entropy[:-1] >= query.entropy[1:])


def test_pseudo_label_threshold_gating() -> None:
    probs = torch.tensor([[0.99], [0.80], [0.04], [0.52]], dtype=torch.float32)
    pseudo = make_pseudo_labels(probs, threshold=0.95)
    assert pseudo.mask.squeeze(-1).tolist() == [True, False, True, False]
    assert pseudo.pseudo_labels.squeeze(-1).tolist() == [1, 1, 0, 1]
