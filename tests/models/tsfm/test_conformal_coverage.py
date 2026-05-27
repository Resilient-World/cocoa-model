"""
Conformal coverage test: asserts TSFM ensemble coverage remains within ±3% of nominal
across a synthetic distribution shift.

Uses the ECI-Integral updater (default) with stratum key
``{scenario}:{horizon}:{region}:tsfm_ensemble``.

Generates a synthetic yield panel with a controlled distribution shift (mean drift
from μ=0.5 to μ=0.35 over 500 time steps) and verifies that the rolling empirical
coverage stays within [0.87, 0.93] for nominal 90% intervals.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from api.online_conformal_store import OnlineConformalStore
from models.tsfm.conformal import TsfmConformalWrapper, tsfm_stratum_key


def _synthetic_shift_panel(
    n_steps: int = 500,
    horizon: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic yield trajectories with a controlled distribution shift.

    Returns
    -------
    (forecasts, observations):
        forecasts: ``[n_steps, horizon]`` synthetic ensemble medians.
        observations: ``[n_steps]`` observed yields (last step of each horizon).
    """
    rng = np.random.default_rng(seed)
    mu_start = 0.5
    mu_end = 0.35
    mu = np.linspace(mu_start, mu_end, n_steps)
    sigma = 0.08

    forecasts = np.zeros((n_steps, horizon), dtype=np.float64)
    observations = np.zeros(n_steps, dtype=np.float64)

    for t in range(n_steps):
        true_mean = mu[t]
        fc = rng.normal(true_mean, sigma * 0.5, horizon)
        forecasts[t] = fc
        obs = rng.normal(true_mean, sigma)
        observations[t] = obs

    return forecasts, observations


def _run_coverage_test(
    forecasts: np.ndarray,
    observations: np.ndarray,
    store: OnlineConformalStore,
    *,
    scenario: str = "ssp245",
    horizon_year: int = 2050,
    region: str = "GHA",
    alpha: float = 0.1,
    burn_in: int = 50,
) -> dict[str, float]:
    """
    Run online conformal updates and compute empirical coverage.

    Parameters
    ----------
    forecasts:
        ``[n_steps, horizon]`` synthetic ensemble medians.
    observations:
        ``[n_steps]`` observed yields.
    store:
        Pre-configured OnlineConformalStore.
    scenario, horizon_year, region:
        Stratum key components.
    alpha:
        Nominal miscoverage rate (0.1 = 90% PI).
    burn_in:
        Number of initial steps to discard before computing coverage.

    Returns
    -------
    dict with ``empirical_coverage``, ``mean_q_t``, ``n_updates``.
    """
    wrapper = TsfmConformalWrapper(store, alpha=alpha)
    n_steps = len(observations)
    covered_count = 0
    q_t_values: list[float] = []

    for t in range(n_steps):
        p50 = forecasts[t]
        obs = float(observations[t])

        result = wrapper.predict_with_conformal(
            scenario=scenario,
            horizon_year=horizon_year,
            region=region,
            p50=p50,
            observed_y=obs,
        )
        q_t_values.append(result["q_t"])

        if t >= burn_in:
            q_adj = result["q_t"]
            lo = p50[-1] - q_adj
            hi = p50[-1] + q_adj
            if lo <= obs <= hi:
                covered_count += 1

    n_eval = n_steps - burn_in
    empirical_coverage = covered_count / n_eval if n_eval > 0 else 0.0
    mean_q_t = float(np.mean(q_t_values[burn_in:])) if n_eval > 0 else 0.0

    return {
        "empirical_coverage": empirical_coverage,
        "mean_q_t": mean_q_t,
        "n_updates": n_steps,
        "n_eval": n_eval,
    }


class TestTsfmConformalCoverage:
    """Assert coverage remains within ±3% of nominal across a distribution shift."""

    def test_coverage_within_tolerance(self) -> None:
        forecasts, observations = _synthetic_shift_panel(n_steps=1000, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "conformal_state.json"
            store = OnlineConformalStore(
                state_path=state_path,
                conformal_method="eci_integral",
                alpha=0.1,
                eci_eta=1.0,
                eci_decay=0.95,
                eci_window=100,
            )

            result = _run_coverage_test(
                forecasts,
                observations,
                store,
                scenario="ssp245",
                horizon_year=2050,
                region="GHA",
                alpha=0.1,
                burn_in=100,
            )

            nominal = 0.9
            coverage = result["empirical_coverage"]
            tolerance = 0.03

            assert abs(coverage - nominal) <= tolerance, (
                f"Coverage {coverage:.4f} outside [{nominal - tolerance:.2f}, "
                f"{nominal + tolerance:.2f}] after {result['n_eval']} updates. "
                f"Mean q_t = {result['mean_q_t']:.4f}"
            )

    def test_coverage_multiple_regions(self) -> None:
        forecasts, observations = _synthetic_shift_panel(n_steps=600, seed=123)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "conformal_state.json"
            store = OnlineConformalStore(
                state_path=state_path,
                conformal_method="eci_integral",
                alpha=0.1,
                eci_eta=1.0,
                eci_decay=0.95,
                eci_window=100,
            )

            for region in ("GHA", "CIV", "CMR"):
                result = _run_coverage_test(
                    forecasts,
                    observations,
                    store,
                    scenario="ssp585",
                    horizon_year=2080,
                    region=region,
                    alpha=0.1,
                    burn_in=100,
                )
                assert abs(result["empirical_coverage"] - 0.9) <= 0.03, (
                    f"Region {region} coverage {result['empirical_coverage']:.4f} out of bounds"
                )

    def test_stratum_key_format(self) -> None:
        key = tsfm_stratum_key("ssp245", 2050, "GHA")
        assert key == "ssp245:2050:GHA:tsfm_ensemble"

        key_corr = tsfm_stratum_key("ssp585", 2080, "CIV", downscaling_method="corrdiff")
        assert key_corr == "ssp585:2080:CIV:tsfm_ensemble:corrdiff"

        key_aurora = tsfm_stratum_key("ssp245", 2030, "CMR", downscaling_method="aurora")
        assert key_aurora == "ssp245:2030:CMR:tsfm_ensemble:aurora"

    def test_conformal_wrapper_no_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "conformal_state.json"
            store = OnlineConformalStore(
                state_path=state_path,
                conformal_method="eci_integral",
                alpha=0.1,
            )
            wrapper = TsfmConformalWrapper(store, alpha=0.1)
            p50 = np.array([0.5, 0.52, 0.48, 0.51], dtype=np.float64)
            result = wrapper.predict_with_conformal(
                scenario="ssp245",
                horizon_year=2050,
                region="GHA",
                p50=p50,
            )
            assert "adjusted_p50" in result
            assert result["q_t"] == 0.0
            assert result["covered"] is None

    def test_conformal_wrapper_with_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "conformal_state.json"
            store = OnlineConformalStore(
                state_path=state_path,
                conformal_method="eci_integral",
                alpha=0.1,
            )
            wrapper = TsfmConformalWrapper(store, alpha=0.1)
            p50 = np.array([0.5, 0.52, 0.48, 0.51], dtype=np.float64)
            result = wrapper.predict_with_conformal(
                scenario="ssp245",
                horizon_year=2050,
                region="GHA",
                p50=p50,
                observed_y=0.5,
            )
            assert result["covered"] is not None
            assert result["stratum_key"] == "ssp245:2050:GHA:tsfm_ensemble"
