"""DiD method comparison harness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.did_comparison_harness import compare_did_methods
from analysis.did_impact import did_estimator


def _staggered_mispecified_dgp(seed: int) -> pd.DataFrame:
    """Staggered DGP where TWFE is biased; CS/BJS/SDID closer to truth."""
    rng = np.random.default_rng(seed)
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
    return pd.DataFrame(rows)


def _true_att(panel: pd.DataFrame) -> float:
    """Cohort-weighted mean treatment effect on treated."""
    effects: list[float] = []
    weights: list[float] = []
    for u, g in panel.groupby("farm_id")["treatment_period"].first().items():
        if np.isnan(g):
            continue
        sub = panel[(panel["farm_id"] == u) & (panel["period"] >= g)]
        pre = panel[(panel["farm_id"] == u) & (panel["period"] < g)]
        if sub.empty or pre.empty:
            continue
        effects.append(float(sub["yield"].mean() - pre["yield"].mean()))
        weights.append(1.0)
    w = np.array(weights)
    w /= w.sum()
    return float(np.dot(w, effects))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_did_methods_columns_and_report(tmp_path) -> None:
    panel = _staggered_mispecified_dgp(1)
    table = compare_did_methods(
        panel,
        methods=["twfe", "csdid", "bjs", "synthdid"],
        n_boot=50,
        n_placebo=30,
        write_report=True,
        out_dir=tmp_path,
    )
    required = {
        "method",
        "ATT",
        "SE",
        "ci_low",
        "ci_high",
        "p_value",
        "pretrend_pvalue",
        "negative_weights_diagnostic",
        "runtime_seconds",
    }
    assert required <= set(table.columns)
    assert len(table) == 4
    reports = list(tmp_path.glob("did_comparison_*.md"))
    assert len(reports) >= 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mse_ranking_twfe_worse_than_robust_estimators() -> None:
    """Monte Carlo: CS ~ BJS ~ SDID outperform TWFE in MSE."""
    n_rep = 40
    mse: dict[str, list[float]] = {k: [] for k in ("twfe", "csdid", "bjs", "synthdid")}
    for seed in range(n_rep):
        panel = _staggered_mispecified_dgp(seed + 100)
        truth = _true_att(panel)
        for method in mse:
            try:
                if method == "twfe":
                    from analysis._staggered_did_common import estimate_twfe, prepare_staggered_panel

                    prep = prepare_staggered_panel(panel)
                    est = estimate_twfe(prep)
                else:
                    res = did_estimator(panel, method=method, n_boot=30, random_state=seed)
                    est = res.att
                mse[method].append((est - truth) ** 2)
            except Exception:
                continue
    means = {k: float(np.mean(v)) for k, v in mse.items() if len(v) >= 10}
    assert means["twfe"] > means["csdid"]
    assert means["twfe"] > means["bjs"]
    assert means["twfe"] > means["synthdid"]
    robust = [means["csdid"], means["bjs"], means["synthdid"]]
    assert max(robust) / min(robust) < 2.5
