#!/usr/bin/env python3
"""
Validate DVDS estimators: Section 7.1 binary DGP, Zhao et al. (2019) bootstrap comparison,
and farm-panel synthetic ATT check.

Writes reports/sensitivity/dvds_validation_<date>.md
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from analysis.dvds import dvds_ate
from data.farm_panel import (
    PSM_COVARIATE_COLS,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_synthetic_panel,
)

_REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = _REPO / "reports" / "sensitivity"


def simulate_section_71_binary(n: int, seed: int) -> pd.DataFrame:
    """Section 7.1 binary-outcome DGP (Dorn, Guo & Kallus 2022)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 * x2
    logit_z = 0.5 * x1 + x2 + 0.5 * x2 * x3
    z = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit_z))).astype(int)
    logit_y = 0.5 * x1 + x2 + 0.25 * x2 * x3
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit_y))).astype(float)
    return pd.DataFrame(
        {
            "received_intervention": z,
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def estimate_true_ate_monte_carlo(n_mc: int = 200_000, seed: int = 0) -> float:
    """Monte Carlo ATE for Section 7.1 DGP under unconfoundedness."""
    df = simulate_section_71_binary(n_mc, seed)
    z = df["received_intervention"].to_numpy()
    y = df["y"].to_numpy()
    return float(y[z == 1].mean() - y[z == 0].mean())


def _crossfit_binary_nuisances(
    df: pd.DataFrame,
    covs: list[str],
    *,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[covs].to_numpy()
    z = df["received_intervention"].to_numpy().astype(int)
    y = df["y"].to_numpy().astype(float)
    n = len(y)
    e_hat = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(x, z):
        g = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        g.fit(x[tr], z[tr])
        e_hat[te] = np.clip(g.predict_proba(x[te])[:, 1], 0.01, 0.99)
        if (z[tr] == 1).sum() > 0:
            m1 = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
            m1.fit(x[tr][z[tr] == 1], y[tr][z[tr] == 1])
            mu1[te] = m1.predict(x[te])
        if (z[tr] == 0).sum() > 0:
            m0 = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
            m0.fit(x[tr][z[tr] == 0], y[tr][z[tr] == 0])
            mu0[te] = m0.predict(x[te])
    return e_hat, mu1, mu0


def zhao_bootstrap_bounds(
    df: pd.DataFrame,
    covs: list[str],
    lambda_: float,
    *,
    n_folds: int = 5,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Zhao et al. (2019) MSM AIPW sensitivity (Section 6.2): plug-in weighting bounds + percentile bootstrap.
    """
    lam = lambda_
    tau = lam / (lam + 1.0)
    e_hat, mu1, mu0 = _crossfit_binary_nuisances(df, covs, n_folds=n_folds, seed=seed)
    z = df["received_intervention"].to_numpy().astype(float)
    y = df["y"].to_numpy().astype(float)
    n = len(y)

    q1_p = (mu1 > 1.0 - tau).astype(float)
    q0_m = (mu0 > tau).astype(float)
    sign1 = np.sign(y - q1_p)
    sign1[sign1 == 0] = 1
    sign0 = np.sign(y - q0_m)
    sign0[sign0 == 0] = 1
    upper_if = z * q1_p / e_hat + (y - q1_p) * z * (1.0 + (1.0 - e_hat) / e_hat * (lam**sign1))
    lower_if = (1.0 - z) * y - (
        (1.0 - z) * q0_m / (1.0 - e_hat)
        + (y - q0_m) * (1.0 - z) * (e_hat / (1.0 - e_hat)) * (lam ** (-sign0))
    )
    ate_upper = float(upper_if.mean() - lower_if.mean())

    q1_m = (mu1 > tau).astype(float)
    q0_p = (mu0 > 1.0 - tau).astype(float)
    sign1m = np.sign(y - q1_m)
    sign1m[sign1m == 0] = 1
    sign0p = np.sign(y - q0_p)
    sign0p[sign0p == 0] = 1
    upper_if_m = z * q1_m / e_hat + (y - q1_m) * z * (
        1.0 + (1.0 - e_hat) / e_hat * (lam ** (-sign1m))
    )
    lower_if_p = (1.0 - z) * y - (
        (1.0 - z) * q0_p / (1.0 - e_hat)
        + (y - q0_p) * (1.0 - z) * (e_hat / (1.0 - e_hat)) * (lam**sign0p)
    )
    ate_lower = float(upper_if_m.mean() - lower_if_p.mean())

    rng = np.random.default_rng(seed)
    boot_upper: list[float] = []
    boot_lower: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_upper.append(float(upper_if[idx].mean() - lower_if[idx].mean()))
        boot_lower.append(float(upper_if_m[idx].mean() - lower_if_p[idx].mean()))

    ci_u = np.percentile(boot_upper, [2.5, 97.5])
    ci_l = np.percentile(boot_lower, [2.5, 97.5])
    return ate_lower, ate_upper, float(ci_l[0]), float(ci_u[1])


def run_binary_dgp_study(
    *,
    n: int,
    n_reps: int,
    lambdas: list[float],
    seed: int,
) -> dict[str, object]:
    true_ate = estimate_true_ate_monte_carlo()
    covs = ["x1", "x2", "x3"]
    point_cover = {lam: 0 for lam in lambdas}
    wald_cover = {lam: 0 for lam in lambdas}
    zhao_width_win = {lam: 0 for lam in lambdas}

    for rep in range(n_reps):
        df = simulate_section_71_binary(n, seed=seed + rep)
        for lam in lambdas:
            res = dvds_ate(
                df,
                treatment_col="received_intervention",
                outcome_col="y",
                covariate_cols=covs,
                lambda_=lam,
                n_folds=5,
                random_state=rep,
            )
            if res.ate_lower <= true_ate <= res.ate_upper:
                point_cover[lam] += 1
            if res.ate_ci_lower <= true_ate <= res.ate_ci_upper:
                wald_cover[lam] += 1

            z_lo, z_hi, z_ci_lo, z_ci_hi = zhao_bootstrap_bounds(
                df, covs, lam, n_bootstrap=200, seed=rep
            )
            dvds_width = res.ate_ci_upper - res.ate_ci_lower
            zhao_width = z_ci_hi - z_ci_lo
            if dvds_width <= zhao_width + 1e-9:
                zhao_width_win[lam] += 1

    return {
        "true_ate": true_ate,
        "n_reps": n_reps,
        "point_cover": {k: v / n_reps for k, v in point_cover.items()},
        "wald_cover": {k: v / n_reps for k, v in wald_cover.items()},
        "dvds_tighter_or_equal": {k: v / n_reps for k, v in zhao_width_win.items()},
    }


def run_farm_panel_study(seed: int = 7) -> dict[str, object]:
    true_att = 0.35
    panel = join_biotic(
        join_climate(load_synthetic_panel(n_farms=800, true_att=true_att, seed=seed))
    )
    snap = farm_level_snapshot(panel, treatment_year=4)
    snap["yield_delta"] = snap["yield_post_intervention"] - snap["yield_pre_intervention"]
    covs = [c for c in PSM_COVARIATE_COLS if c in snap.columns]
    widths: dict[float, float] = {}
    contains_att: dict[float, bool] = {}
    for lam in [1.1, 1.5, 2.0, 3.0]:
        res = dvds_ate(
            snap.dropna(subset=["yield_delta", *covs]),
            treatment_col="received_intervention",
            outcome_col="yield_delta",
            covariate_cols=covs,
            lambda_=lam,
            n_folds=5,
            random_state=seed,
        )
        widths[lam] = res.ate_upper - res.ate_lower
        contains_att[lam] = res.ate_lower <= true_att <= res.ate_upper
    return {"true_att": true_att, "widths": widths, "contains_att": contains_att}


def write_report(path: Path, binary: dict, farm: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DVDS validation report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Section 7.1 binary DGP (n=1000)",
        "",
        f"Monte Carlo true ATE: **{binary['true_ate']:.4f}** ({binary['n_reps']} replications)",
        "",
        "| Λ | Point bound coverage | Wald CI coverage | DVDS width ≤ Zhao bootstrap |",
        "|---|---------------------|------------------|----------------------------|",
    ]
    for lam in sorted(binary["point_cover"]):
        lines.append(
            f"| {lam} | {binary['point_cover'][lam]:.1%} | {binary['wald_cover'][lam]:.1%} | "
            f"{binary['dvds_tighter_or_equal'][lam]:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Farm panel synthetic ATT",
            "",
            f"True ATT (tonnes/ha): **{farm['true_att']:.2f}**",
            "",
            "| Λ | Interval width | Contains true ATT |",
            "|---|----------------|-------------------|",
        ]
    )
    for lam, w in sorted(farm["widths"].items()):
        ok = "yes" if farm["contains_att"][lam] else "no"
        lines.append(f"| {lam} | {w:.4f} | {ok} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DVDS sensitivity estimators")
    parser.add_argument("--n", type=int, default=1000, help="Sample size per replication")
    parser.add_argument(
        "--reps", type=int, default=50, help="Monte Carlo replications (use 200+ for full gate)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = run_binary_dgp_study(
        n=args.n,
        n_reps=args.reps,
        lambdas=[1.0, 1.5, 2.0, 2.5],
        seed=args.seed,
    )
    farm = run_farm_panel_study(seed=args.seed)

    out = args.out or REPORT_DIR / f"dvds_validation_{date.today().isoformat()}.md"
    write_report(out, binary, farm)
    print(f"Wrote {out}")
    print("Binary Wald coverage:", binary["wald_cover"])
    print("Farm panel contains ATT at Λ=2:", farm["contains_att"].get(2.0))


if __name__ == "__main__":
    main()
