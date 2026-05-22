"""HTML report templates for causal ATT pipeline outputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.did_impact import DiDResult
from analysis.psm_matching import AIPWResult, BalanceReport
from analysis.sensitivity import EValueResult


def _float_fmt(value: float) -> str:
    return f"{value:.4f}"


def _df_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    html = df.head(max_rows).to_html(index=False, float_format=_float_fmt)
    return str(html)


def att_report_html(
    *,
    aipw: AIPWResult,
    did: DiDResult,
    balance: BalanceReport,
    att_agreement_ok: bool,
    att_agreement_delta: float,
    rosenbaum: pd.DataFrame,
    evalue: EValueResult,
    panel_summary: dict[str, Any],
    true_att: float | None = None,
) -> str:
    """Build a self-contained HTML ATT report."""
    status = "PASS" if balance.balance_ok and att_agreement_ok else "FAIL"
    true_row = ""
    if true_att is not None:
        true_row = f"<tr><td>Known synthetic ATT</td><td>{true_att:.3f} t/ha</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ATT Report — {date.today().isoformat()}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 960px; }}
    h1, h2 {{ color: #1a3a2a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #e8f5e9; }}
    .pass {{ color: #2e7d32; font-weight: bold; }}
    .fail {{ color: #c62828; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Cocoa farm panel — ATT causal report</h1>
  <p>Generated {date.today().isoformat()}</p>
  <p>Overall gate: <span class="{"pass" if status == "PASS" else "fail"}">{status}</span></p>

  <h2>Panel summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Farms</td><td>{panel_summary.get("n_farms", "—")}</td></tr>
    <tr><td>Farm-years</td><td>{panel_summary.get("n_rows", "—")}</td></tr>
    <tr><td>Treated farms</td><td>{panel_summary.get("n_treated_farms", "—")}</td></tr>
    {true_row}
  </table>

  <h2>Balance (PSM)</h2>
  <p>Max |SMD| matched: {balance.max_smd_matched:.4f}
     (threshold 0.10) — {"OK" if balance.balance_ok else "FAIL"}</p>
  {_df_html(balance.smd)}

  <h2>AIPW / DML (cross-fit)</h2>
  <table>
    <tr><th>Estimand</th><th>Point</th><th>SE</th><th>95% CI</th></tr>
    <tr><td>ATT</td><td>{aipw.att:.4f}</td><td>{aipw.att_se:.4f}</td>
        <td>[{aipw.att_ci_low:.4f}, {aipw.att_ci_high:.4f}]</td></tr>
    <tr><td>ATE</td><td>{aipw.ate:.4f}</td><td>{aipw.ate_se:.4f}</td>
        <td>[{aipw.ate_ci_low:.4f}, {aipw.ate_ci_high:.4f}]</td></tr>
  </table>

  <h2>DiD (matched pairs, bootstrap)</h2>
  <table>
    <tr><th>ATT</th><th>SE</th><th>95% CI</th><th>Pairs</th></tr>
    <tr><td>{did.att:.4f}</td><td>{did.se if did.se is not None else float("nan"):.4f}</td>
        <td>[{did.ci_low}, {did.ci_high}]</td><td>{did.n_pairs}</td></tr>
  </table>

  <h2>Cross-check</h2>
  <p>|AIPW ATT − DiD ATT| = {att_agreement_delta:.4f};
     within 1 SE: {"yes" if att_agreement_ok else "no"}</p>

  <h2>Sensitivity</h2>
  <p>E-value (point): {evalue.point_e_value:.2f}; E-value (CI bound): {evalue.ci_e_value:.2f}</p>
  <h3>Rosenbaum bounds</h3>
  {_df_html(rosenbaum)}
</body>
</html>
"""


def write_att_report(
    path: Path,
    *,
    aipw: AIPWResult,
    did: DiDResult,
    balance: BalanceReport,
    att_agreement_ok: bool,
    att_agreement_delta: float,
    rosenbaum: pd.DataFrame,
    evalue: EValueResult,
    panel_summary: dict[str, Any],
    true_att: float | None = None,
) -> Path:
    """Write ``ATT_report_<date>.html`` to ``path`` (file or directory)."""
    if path.suffix.lower() != ".html":
        path = path / f"ATT_report_{date.today().isoformat()}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        att_report_html(
            aipw=aipw,
            did=did,
            balance=balance,
            att_agreement_ok=att_agreement_ok,
            att_agreement_delta=att_agreement_delta,
            rosenbaum=rosenbaum,
            evalue=evalue,
            panel_summary=panel_summary,
            true_att=true_att,
        ),
        encoding="utf-8",
    )
    return path


__all__ = ["att_report_html", "write_att_report"]
