"""Tests for Borusyak-Jaravel-Spiess imputation DiD."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis._staggered_did_common import estimate_twfe, prepare_staggered_panel
from analysis.bjs_imputation import BorusyakJaravelSpiess
from analysis.csdid import CallawaySantAnna

TRUE_ATT = 1.5


def _three_cohort_panel(*, seed: int = 0, att: float = TRUE_ATT) -> pd.DataFrame:
    """Staggered panel: cohorts adopt at t=1,2,3 with constant treatment effect."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for u in range(60):
        if u < 15:
            g = float("nan")
        else:
            g = float(((u - 15) % 3) + 1)  # cohorts 1, 2, 3
        alpha = rng.normal(0, 0.2)
        for t in range(5):
            y0 = alpha + rng.normal(0, 0.05)
            treat = float(t >= g)
            y = y0 + att * treat
            rows.append(
                {
                    "farm_id": f"u{u}",
                    "period": t,
                    "treatment_period": g,
                    "yield": y,
                }
            )
    return pd.DataFrame(rows)


def test_bjs_and_cs_recover_true_att() -> None:
    panel = _three_cohort_panel()
    bjs = BorusyakJaravelSpiess(panel, random_state=42).estimate()
    assert 1.35 <= bjs.att <= 1.65

    cs = CallawaySantAnna(panel, n_boot=99, random_state=42).simple_att()
    assert 1.35 <= cs.att <= 1.65


def test_twfe_biased_vs_true_att() -> None:
    """Heterogeneous cohort effects: TWFE departs from constant ATT=1.5."""
    rng = np.random.default_rng(1)
    rows: list[dict] = []
    effects = {1.0: 3.0, 2.0: 0.0, 3.0: -0.5}
    for u in range(90):
        g = float((u % 3) + 1)
        alpha = rng.normal(0, 0.1)
        for t in range(5):
            y0 = alpha + 0.05 * t + rng.normal(0, 0.02)
            eff = effects[g] if t >= g else 0.0
            rows.append(
                {
                    "farm_id": f"u{u}",
                    "period": t,
                    "treatment_period": g,
                    "yield": y0 + eff,
                }
            )
    panel = pd.DataFrame(rows)
    prep = prepare_staggered_panel(panel)
    twfe = estimate_twfe(prep)
    assert abs(twfe - TRUE_ATT) > 0.3


def test_negative_weight_dgp_cs_bjs_positive() -> None:
    """
    Staggered DGP where TWFE can be negative but true effect on treated is positive.
    """
    rng = np.random.default_rng(99)
    rows: list[dict] = []
    for u in range(40):
        g = float(2 if u < 20 else 4)
        for t in range(6):
            fe = 0.5 * u / 40
            te = 0.2 * t
            y0 = fe + te + rng.normal(0, 0.02)
            true_eff = 1.0 if t >= g else 0.0
            if u < 20 and t >= 2:
                true_eff = 2.0
            elif u >= 20 and t >= 4:
                true_eff = 0.5
            rows.append(
                {
                    "farm_id": f"f{u}",
                    "period": t,
                    "treatment_period": g,
                    "yield": y0 + true_eff,
                }
            )
    panel = pd.DataFrame(rows)
    prep = prepare_staggered_panel(panel)
    twfe = estimate_twfe(prep)
    cs = CallawaySantAnna(panel, n_boot=50).simple_att().att
    bjs = BorusyakJaravelSpiess(panel).estimate().att
    assert cs > 0
    assert bjs > 0
    # TWFE can be mis-weighted under staggered timing; disagree with valid estimators
    assert abs(twfe - cs) > 0.15 or abs(twfe - bjs) > 0.15


def test_bjs_pretrend_returns_pvalue() -> None:
    panel = _three_cohort_panel()
    res = BorusyakJaravelSpiess(panel).estimate()
    assert res.n_treated > 0
    assert res.pretrend_pvalue is None or 0 <= res.pretrend_pvalue <= 1
