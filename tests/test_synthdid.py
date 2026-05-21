"""Synthetic DiD replication (California Prop 99)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.synthdid import SyntheticDiD, synthdid_att

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthdid" / "california_prop99.csv"


def _prop99_long_panel() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE, sep=";")
    rows: list[dict] = []
    for state, grp in raw.groupby("State"):
        g = 1989.0 if grp["treated"].max() == 1 else np.nan
        for _, r in grp.iterrows():
            rows.append(
                {
                    "farm_id": state,
                    "period": int(r["Year"]),
                    "treatment_period": g,
                    "yield": float(r["PacksPerCapita"]),
                }
            )
    return pd.DataFrame(rows)


def _prop99_matrix() -> tuple[np.ndarray, int, int]:
    raw = pd.read_csv(FIXTURE, sep=";")
    raw = raw.sort_values(["State", "Year"])
    years = sorted(raw["Year"].unique())
    Y = raw.pivot_table(
        index="State", columns="Year", values="PacksPerCapita", aggfunc="first"
    ).to_numpy()
    w = raw.groupby("State")["treated"].max().values
    T0 = next(i for i, y in enumerate(years) if raw.loc[raw["Year"] == y, "treated"].any())
    order = list(np.where(w == 0)[0]) + list(np.where(w == 1)[0])
    Y = Y[order]
    N0 = int((w == 0).sum())
    return Y, N0, T0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_prop99_att_near_minus_19() -> None:
    """Arkhangelsky et al. (2021) Prop 99 ATT ~ -19 packs/capita."""
    Y, N0, T0 = _prop99_matrix()
    att, _, _ = synthdid_att(Y, N0, T0)
    assert -20.0 <= att <= -18.0, f"ATT={att}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_synthetic_did_class_prop99() -> None:
    panel = _prop99_long_panel()
    res = SyntheticDiD(
        panel,
        unit_col="farm_id",
        time_col="period",
        treat_time_col="treatment_period",
        outcome_col="yield",
        n_placebo=50,
        random_state=0,
    ).estimate()
    assert -21.0 <= res.att <= -17.0
    assert res.n_treated >= 1
    assert res.n_control >= 10


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cvxpy_simplex_solver() -> None:
    """cvxpy QP path is available for simplex weights."""
    from analysis.synthdid import _solve_simplex_weights_cvxpy

    A = np.eye(3)
    target = np.array([1.0, 0.0, 0.0])
    w = _solve_simplex_weights_cvxpy(A, target, zeta=1e-4)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()
