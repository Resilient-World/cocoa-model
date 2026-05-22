#!/usr/bin/env python3
"""
Generate tests/fixtures/conformal/*.npz for Wu et al. (2025) Table 1/2 style coverage tests.

Uses synthetic Prophet-like score streams (finance-domain protocol) when stock CSV
is unavailable. Re-run to refresh fixtures after changing score generator.

Example::

    python scripts/export_conformal_replication_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def synthetic_prophet_like_scores(n: int, *, seed: int, vol: float = 0.4) -> np.ndarray:
    """Exponential scores with mild AR persistence (finance-like residuals)."""
    rng = np.random.default_rng(seed)
    base = rng.exponential(scale=1.0 / max(vol, 0.1), size=n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.05)
    return np.maximum(base + ar, 0.0).astype(np.float64)


OUT = _REPO / "tests" / "fixtures" / "conformal"
N = 2000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, seed, vol in [
        ("amazon_prophet_scores", 42, 0.38),
        ("google_prophet_scores", 43, 0.42),
    ]:
        scores = synthetic_prophet_like_scores(N, seed=seed, vol=vol)
        path = OUT / f"{name}.npz"
        np.savez(
            path,
            scores=scores,
            alpha=np.array(0.1),
            source="synthetic_prophet_like; align with Wu et al. 2025 ICLR finance protocol",
            n=N,
        )
        print(f"Wrote {path} ({len(scores)} scores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
