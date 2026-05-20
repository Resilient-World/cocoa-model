#!/usr/bin/env python3
"""
Generate a one-page PDF causal report (matplotlib) for the farm panel pipeline.

Includes: Love plot, common-support density, AIPW vs DiD ATT, Rosenbaum Γ curve,
E-value annotation, and event-study coefficients.

Example::

    python scripts/generate_causal_report.py --synthetic
    python scripts/generate_causal_report.py --panel data/raw/farm_panel.parquet \\
        --out reports/causal/causal_report.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from analysis.did_impact import calculate_did_att, event_study
from analysis.parallel_trends import goodman_bacon_decomposition, placebo_pretreatment_did
from analysis.psm_matching import (
    aipw_estimator,
    compute_propensity_scores,
    default_logit_caliper,
    love_plot_data,
    match_nearest_neighbor,
    standardized_mean_differences,
    trim_common_support,
)
from analysis.sensitivity import (
    e_value,
    negative_control_outcome_test,
    rosenbaum_bounds,
    rosenbaum_gamma_at_alpha,
)
from data.farm_panel import (
    PSM_COVARIATE_COLS,
    attach_pre_post_to_matched,
    farm_level_snapshot,
    join_biotic,
    join_climate,
    load_real_panel,
    load_synthetic_panel,
    treatment_year_index,
)

logger = logging.getLogger(__name__)
DEFAULT_ERA5 = _REPO_ROOT / "data" / "processed" / "era5_2024.zarr"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "generate_causal_report requires matplotlib. "
            "Install with: pip install -e '.[dev]'"
        ) from exc
    return plt


def _panel_to_event_study(panel: pd.DataFrame, treatment_year: int) -> pd.DataFrame:
    years = sorted(panel["year"].unique())
    split = years[treatment_year]
    long = panel.copy()
    long["period"] = long["year"] - split
    long["treatment_period"] = np.where(long["received_intervention"] == 1, 0, np.nan)
    return long.rename(columns={"yield_tonnes_per_ha": "yield"})


def build_report_context(
    panel: pd.DataFrame,
    *,
    treatment_year: int | None = None,
    era5_zarr_path: Path | None = None,
    random_state: int = 42,
) -> dict:
    """Run PSM, AIPW, DiD, sensitivity, and parallel-trends; return plot inputs."""
    panel = join_climate(panel, era5_zarr_path)
    panel = join_biotic(panel)
    if treatment_year is None:
        treatment_year = treatment_year_index(panel)

    snapshot = farm_level_snapshot(panel, treatment_year=treatment_year)
    covariates = [c for c in PSM_COVARIATE_COLS if c in snapshot.columns]

    work = snapshot.copy()
    work["propensity_score"] = compute_propensity_scores(
        work, covariate_cols=covariates, random_state=random_state
    )
    work_trim = trim_common_support(work)
    caliper = default_logit_caliper(work_trim["propensity_score"].to_numpy())
    matched = match_nearest_neighbor(
        work_trim, k=1, caliper=caliper, caliper_scale="logit"
    )
    balance = standardized_mean_differences(
        snapshot, matched, covariate_cols=covariates
    )
    matched_did = attach_pre_post_to_matched(matched, snapshot)

    aipw = aipw_estimator(
        snapshot,
        outcome_col="yield_tonnes_per_ha",
        covariate_cols=covariates,
        n_folds=5,
        random_state=random_state,
    )
    did = calculate_did_att(matched_did, random_state=random_state)
    rosenbaum = rosenbaum_bounds(matched_did, outcome_col=None)
    gamma_star = rosenbaum_gamma_at_alpha(rosenbaum, alpha=0.05)
    ev = e_value(aipw.att, aipw.att_se, outcome_sd=float(snapshot["yield_tonnes_per_ha"].std(ddof=1)))

    nco = None
    if "soil_quality_index" in snapshot.columns:
        nco = negative_control_outcome_test(snapshot, "soil_quality_index")

    placebo = placebo_pretreatment_did(
        panel,
        treatment_year=sorted(panel["year"].unique())[treatment_year],
        k_periods=3,
        random_state=random_state,
    )
    bacon = goodman_bacon_decomposition(panel)
    es_panel = _panel_to_event_study(panel, treatment_year)
    try:
        event_res = event_study(es_panel)
    except ImportError:
        event_res = None

    return {
        "snapshot": snapshot,
        "work_trim": work_trim,
        "matched": matched,
        "matched_did": matched_did,
        "balance": balance,
        "love": love_plot_data(balance),
        "aipw": aipw,
        "did": did,
        "rosenbaum": rosenbaum,
        "gamma_star": gamma_star,
        "evalue": ev,
        "nco": nco,
        "placebo": placebo,
        "bacon": bacon,
        "event_study": event_res,
        "treatment_year": treatment_year,
    }


def write_causal_report_pdf(ctx: dict, out_path: Path) -> Path:
    """Render a single-page PDF causal summary."""
    plt = _require_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"Cocoa farm panel — causal report ({date.today().isoformat()})", fontsize=12)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Love plot
    ax_love = fig.add_subplot(gs[0, 0])
    love = ctx["love"]
    for stage, marker, color in [("before", "o", "#c62828"), ("after", "s", "#2e7d32")]:
        sub = love[love["stage"] == stage]
        ax_love.scatter(sub["smd"], sub["covariate"], label=stage, marker=marker, color=color)
    ax_love.axvline(0.1, color="gray", ls="--", lw=0.8)
    ax_love.axvline(-0.1, color="gray", ls="--", lw=0.8)
    ax_love.set_xlabel("|SMD|")
    ax_love.set_title("Love plot")
    ax_love.legend(fontsize=7)

    # Common support
    ax_ps = fig.add_subplot(gs[0, 1])
    ps = ctx["work_trim"]
    ax_ps.hist(
        ps.loc[ps["received_intervention"] == 1, "propensity_score"],
        bins=25,
        alpha=0.5,
        density=True,
        label="treated",
        color="#1565c0",
    )
    ax_ps.hist(
        ps.loc[ps["received_intervention"] == 0, "propensity_score"],
        bins=25,
        alpha=0.5,
        density=True,
        label="control",
        color="#ef6c00",
    )
    ax_ps.set_xlabel("Propensity score")
    ax_ps.set_title("Common support")
    ax_ps.legend(fontsize=7)

    # ATT comparison
    ax_att = fig.add_subplot(gs[0, 2])
    aipw, did = ctx["aipw"], ctx["did"]
    labels = ["AIPW ATT", "DiD ATT"]
    points = [aipw.att, did.att]
    errs = [1.96 * aipw.att_se, 1.96 * (did.se or aipw.att_se)]
    ax_att.errorbar(labels, points, yerr=errs, fmt="o", capsize=5, color="#1a3a2a")
    ax_att.set_ylabel("t/ha")
    ax_att.set_title("ATT ± 95% CI")
    ax_att.axhline(0, color="gray", lw=0.6)

    # Rosenbaum
    ax_ros = fig.add_subplot(gs[1, 0])
    rb = ctx["rosenbaum"]
    ax_ros.plot(rb["gamma"], rb["p_value_upper"], "o-", ms=3)
    ax_ros.axhline(0.05, color="gray", ls="--", lw=0.8)
    g_star = ctx["gamma_star"]
    title_ros = "Rosenbaum bounds"
    if g_star is not None:
        title_ros += f"\nΓ* (p>0.05) ≈ {g_star:.2f}"
    else:
        title_ros += "\nΓ* > grid max"
    ax_ros.set_xlabel("Γ")
    ax_ros.set_ylabel("p-value (upper)")
    ax_ros.set_title(title_ros, fontsize=9)

    # E-value text panel
    ax_ev = fig.add_subplot(gs[1, 1])
    ax_ev.axis("off")
    ev = ctx["evalue"]
    lines = [
        "E-value (VanderWeele & Ding 2017)",
        f"Point: {ev.point_e_value:.2f}",
        f"CI bound: {ev.ci_e_value:.2f}",
        f"ATT: {ev.estimate:.3f} t/ha",
        f"CI low: {ev.ci_low:.3f}",
    ]
    nco = ctx.get("nco")
    if nco is not None:
        lines.append(f"NCO ({nco.nco_col}): p={nco.p_value:.3f} ({'PASS' if nco.falsification_pass else 'FAIL'})")
    pb = ctx["placebo"]
    lines.append(f"Placebo pre-trends OK: {pb.joint_pretrend_ok}")
    ax_ev.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")

    # Placebo pre-treatment
    ax_pl = fig.add_subplot(gs[1, 2])
    pt = pb.table
    if not pt.empty and pt["placebo_att"].notna().any():
        ax_pl.errorbar(
            pt["k"],
            pt["placebo_att"],
            yerr=[
                pt["placebo_att"] - pt["ci_low"],
                pt["ci_high"] - pt["placebo_att"],
            ],
            fmt="o",
            capsize=3,
        )
    ax_pl.axhline(0, color="gray", lw=0.6)
    ax_pl.set_xlabel("k periods before treatment")
    ax_pl.set_ylabel("Placebo ATT")
    ax_pl.set_title("Pre-treatment placebo DiD")

    # Event study (span bottom row)
    ax_es = fig.add_subplot(gs[2, :])
    es = ctx["event_study"]
    if es is not None and not es.leads_lags.empty:
        ll = es.leads_lags
        ax_es.errorbar(
            ll["period"],
            ll["coef"],
            yerr=[ll["coef"] - ll["ci_low"], ll["ci_high"] - ll["coef"]],
            fmt="o-",
            capsize=3,
        )
        pretrend = es.pretrend_pvalue
        ax_es.set_title(
            f"Event study (pre-trend F-test p={pretrend:.3f})"
            if pretrend is not None
            else "Event study"
        )
    else:
        ax_es.text(0.5, 0.5, "Event study unavailable\n(install statsmodels)", ha="center", va="center")
        ax_es.set_title("Event study")
    ax_es.axhline(0, color="gray", lw=0.6)
    ax_es.set_xlabel("Event time (periods relative to treatment)")

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-page PDF causal report")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--n-farms", type=int, default=2000)
    parser.add_argument("--n-years", type=int, default=8)
    parser.add_argument("--treatment-year", type=int, default=4)
    parser.add_argument("--true-att", type=float, default=0.35)
    parser.add_argument("--era5-zarr", type=Path, default=DEFAULT_ERA5)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "reports" / "causal" / f"causal_report_{date.today().isoformat()}.pdf",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.synthetic:
        panel = load_synthetic_panel(
            n_farms=args.n_farms,
            n_years=args.n_years,
            treatment_year=args.treatment_year,
            true_att=args.true_att,
            seed=args.seed,
        )
        treatment_year = args.treatment_year
    elif args.panel is not None:
        panel = load_real_panel(args.panel)
        treatment_year = args.treatment_year
    else:
        parser.error("Specify --synthetic or --panel PATH")

    era5 = args.era5_zarr if args.era5_zarr.is_dir() else None
    ctx = build_report_context(
        panel,
        treatment_year=treatment_year,
        era5_zarr_path=era5,
        random_state=args.seed,
    )
    path = write_causal_report_pdf(ctx, args.out)
    logger.info("Wrote causal PDF report to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
