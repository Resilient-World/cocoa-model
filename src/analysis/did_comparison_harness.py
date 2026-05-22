"""
Compare TWFE, Callaway-Sant'Anna, BJS, and Synthetic DiD on the same panel.

Writes a markdown report with a forest plot under ``reports/causal/``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis._staggered_did_common import normal_ci, prepare_staggered_panel
from analysis.bjs_imputation import BorusyakJaravelSpiess
from analysis.csdid import CallawaySantAnna
from analysis.parallel_trends import goodman_bacon_decomposition, placebo_pretreatment_did
from analysis.synthdid import SyntheticDiD

METHOD_ALIASES = {
    "twfe": "twfe",
    "cs": "csdid",
    "csdid": "csdid",
    "bjs": "bjs",
    "sdid": "synthdid",
    "synthdid": "synthdid",
}


def _p_from_z(z: float) -> float:
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))))


def _build_did_panel(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
) -> pd.DataFrame:
    """Add ``received_intervention`` for Goodman-Bacon / placebo tests."""
    work = df.copy()
    if "_G" in work.columns:
        g = work["_G"]
    else:
        g = work.groupby(unit_col)[treat_time_col].min()
        work = work.merge(g.rename("_G"), on=unit_col, how="left")
        g = work["_G"]
    t = work[time_col]
    work["received_intervention"] = ((t >= g) & g.notna()).astype(int)
    return work


def _negative_weight_diagnostic(
    panel: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
) -> float:
    """Share of TWFE weight from already-treated controls (Goodman-Bacon)."""
    bac = goodman_bacon_decomposition(
        panel,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col="received_intervention",
    )
    if bac.empty or "weight_share" not in bac.columns:
        return 0.0
    forbidden = bac[bac["comparison"] == "timing_vs_already_treated"]
    return float(forbidden["weight_share"].sum()) if len(forbidden) else 0.0


def _estimate_twfe_row(
    prep: Any,
    *,
    alpha: float,
) -> dict[str, Any]:
    from linearmodels.panel import PanelOLS

    work = prep.df.copy()
    g = work["_G"]
    t = work[prep.time_col]
    d_it = ((t >= g) & g.notna()).astype(float)
    work["_D_it"] = d_it
    work = work.set_index([prep.unit_col, prep.time_col])
    y = work[prep.outcome_col]
    if prep.covariate_cols:
        exog = pd.concat([work[["_D_it"]], work[prep.covariate_cols]], axis=1)
        mod = PanelOLS(y, exog, entity_effects=True, time_effects=True)
    else:
        mod = PanelOLS(y, work[["_D_it"]], entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    att = float(res.params["_D_it"])
    se = float(res.std_errors["_D_it"])
    lo, hi = normal_ci(att, se, alpha)
    p = _p_from_z(att / se) if se > 0 else None

    panel_did = _build_did_panel(
        prep.df,
        unit_col=prep.unit_col,
        time_col=prep.time_col,
        treat_time_col=prep.treat_time_col,
    )
    pretreat = None
    cohorts = prep.cohorts
    if len(cohorts):
        g0 = float(cohorts[0])
        try:
            pb = placebo_pretreatment_did(
                panel_did,
                treatment_year=int(g0),
                unit_col=prep.unit_col,
                time_col=prep.time_col,
                outcome_col=prep.outcome_col,
                treat_col="received_intervention",
            )
            pretreat = 1.0 if pb.joint_pretrend_ok else 0.01
        except (ValueError, TypeError):
            pretreat = None

    return {
        "ATT": att,
        "SE": se,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "pretrend_pvalue": pretreat,
    }


def _estimate_csdid_row(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str] | None,
    n_boot: int,
    alpha: float,
    random_state: int,
) -> dict[str, Any]:
    est = CallawaySantAnna(
        df,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        n_boot=n_boot,
        alpha=alpha,
        random_state=random_state,
    )
    res = est.simple_att()
    es = est.event_study_aggregation()
    pretrend = None
    if es.leads_lags is not None and not es.leads_lags.empty:
        pre = es.leads_lags[es.leads_lags["event_time"] < 0]
        if len(pre) and "att" in pre.columns:
            pretrend = float((pre["att"].abs() < 1e-6).mean())
    p = _p_from_z(res.att / res.se) if res.se and res.se > 0 else None
    return {
        "ATT": res.att,
        "SE": res.se,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p_value": p,
        "pretrend_pvalue": pretrend,
    }


def _estimate_bjs_row(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str] | None,
    alpha: float,
    random_state: int,
) -> dict[str, Any]:
    res = BorusyakJaravelSpiess(
        df,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        random_state=random_state,
    ).estimate()
    p = _p_from_z(res.att / res.se) if res.se and res.se > 0 else None
    return {
        "ATT": res.att,
        "SE": res.se,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p_value": p,
        "pretrend_pvalue": res.pretrend_pvalue,
    }


def _estimate_synthdid_row(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
    outcome_col: str,
    n_placebo: int,
    alpha: float,
    random_state: int,
) -> dict[str, Any]:
    est = SyntheticDiD(
        df,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
        outcome_col=outcome_col,
        n_placebo=n_placebo,
        alpha=alpha,
        random_state=random_state,
    )
    res = est.estimate()
    return {
        "ATT": res.att,
        "SE": res.se,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p_value": res.p_value,
        "pretrend_pvalue": est.pretrend_placebo_pvalue(),
    }


def compare_did_methods(
    df: pd.DataFrame,
    methods: Sequence[str] = ("twfe", "csdid", "bjs", "synthdid"),
    *,
    unit_col: str = "farm_id",
    time_col: str = "period",
    treat_time_col: str = "treatment_period",
    outcome_col: str = "yield",
    covariate_cols: Sequence[str] | None = None,
    n_boot: int = 199,
    n_placebo: int = 100,
    alpha: float = 0.05,
    random_state: int = 42,
    write_report: bool = True,
    out_dir: Path | str = "reports/causal",
) -> pd.DataFrame:
    """
    Run DiD estimators on the same panel and return a comparison table.

    Columns: method, ATT, SE, ci_low, ci_high, p_value, pretrend_pvalue,
    negative_weights_diagnostic, runtime_seconds.
    """
    prep = prepare_staggered_panel(
        df,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
    )
    panel_did = _build_did_panel(
        prep.df,
        unit_col=unit_col,
        time_col=time_col,
        treat_time_col=treat_time_col,
    )
    neg_diag = _negative_weight_diagnostic(
        panel_did,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )

    rows: list[dict[str, Any]] = []
    for raw in methods:
        key = METHOD_ALIASES.get(raw.lower(), raw.lower())
        t0 = time.perf_counter()
        try:
            if key == "twfe":
                stats = _estimate_twfe_row(prep, alpha=alpha)
            elif key == "csdid":
                stats = _estimate_csdid_row(
                    df,
                    unit_col=unit_col,
                    time_col=time_col,
                    treat_time_col=treat_time_col,
                    outcome_col=outcome_col,
                    covariate_cols=covariate_cols,
                    n_boot=n_boot,
                    alpha=alpha,
                    random_state=random_state,
                )
            elif key == "bjs":
                stats = _estimate_bjs_row(
                    df,
                    unit_col=unit_col,
                    time_col=time_col,
                    treat_time_col=treat_time_col,
                    outcome_col=outcome_col,
                    covariate_cols=covariate_cols,
                    alpha=alpha,
                    random_state=random_state,
                )
            elif key == "synthdid":
                stats = _estimate_synthdid_row(
                    df,
                    unit_col=unit_col,
                    time_col=time_col,
                    treat_time_col=treat_time_col,
                    outcome_col=outcome_col,
                    n_placebo=n_placebo,
                    alpha=alpha,
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unknown method: {raw!r}")
        except Exception as exc:
            stats = {
                "ATT": float("nan"),
                "SE": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_value": None,
                "pretrend_pvalue": None,
                "error": str(exc),
            }
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "method": key,
                "ATT": stats["ATT"],
                "SE": stats["SE"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "p_value": stats.get("p_value"),
                "pretrend_pvalue": stats.get("pretrend_pvalue"),
                "negative_weights_diagnostic": neg_diag,
                "runtime_seconds": elapsed,
            }
        )

    table = pd.DataFrame(rows)
    if write_report:
        write_did_comparison_report(table, out_dir=out_dir)
    return table


def write_did_comparison_report(
    table: pd.DataFrame,
    *,
    out_dir: Path | str = "reports/causal",
) -> Path:
    """Write markdown report and optional forest plot PNG."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    md_path = out_path / f"did_comparison_{today}.md"
    png_path = out_path / f"did_comparison_{today}_forest.png"

    display = table.copy()
    display["95% CI"] = display.apply(
        lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if pd.notna(r["ci_low"]) else "NA",
        axis=1,
    )
    cols = [
        "method",
        "ATT",
        "SE",
        "95% CI",
        "p_value",
        "pretrend_pvalue",
        "negative_weights_diagnostic",
        "runtime_seconds",
    ]
    md_lines = [
        f"# DiD method comparison ({today})",
        "",
        "TWFE vs Callaway-Sant'Anna vs BJS vs Synthetic DiD on the same panel.",
        "",
        "```\n" + display[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n```",
        "",
    ]

    _try_forest_plot(table, png_path)
    if png_path.is_file():
        md_lines.append(f"![Forest plot]({png_path.name})")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return md_path


def _try_forest_plot(table: pd.DataFrame, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = table.dropna(subset=["ATT", "ci_low", "ci_high"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(valid))))
    y_pos = np.arange(len(valid))
    att = valid["ATT"].to_numpy()
    lo = valid["ci_low"].to_numpy()
    hi = valid["ci_high"].to_numpy()
    err_lo = att - lo
    err_hi = hi - att
    ax.errorbar(
        att,
        y_pos,
        xerr=[err_lo, err_hi],
        fmt="o",
        capsize=4,
        color="steelblue",
    )
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid["method"].tolist())
    ax.set_xlabel("ATT")
    ax.set_title("DiD method comparison (95% CI)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


__all__ = ["METHOD_ALIASES", "compare_did_methods", "write_did_comparison_report"]
