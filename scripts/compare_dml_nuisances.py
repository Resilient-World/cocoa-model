#!/usr/bin/env python3
"""Compare HGB vs NGBoost AIPW on synthetic heteroscedastic panel."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import importlib.util

_psm_path = _REPO_ROOT / "src" / "analysis" / "psm_matching.py"
_psm_spec = importlib.util.spec_from_file_location("analysis.psm_matching", _psm_path)
assert _psm_spec and _psm_spec.loader
_psm = importlib.util.module_from_spec(_psm_spec)
sys.modules["analysis.psm_matching"] = _psm
_psm_spec.loader.exec_module(_psm)
aipw_estimator = _psm.aipw_estimator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "reports" / "causal")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(42)
    n = 1200
    x1 = rng.normal(size=n)
    treat = (rng.uniform(size=n) < 0.4).astype(int)
    y = 1.5 + 0.35 * treat + rng.normal(scale=np.exp(0.4 * x1))
    df = pd.DataFrame({"received_intervention": treat, "yield_delta": y, "x1": x1})
    rows = []
    for est in ("hgb", "ngboost"):
        try:
            res = aipw_estimator(
                df, outcome_col="yield_delta", covariate_cols=["x1"], nuisance_estimator=est
            )
            rows.append((est, res.ate, res.ate_se, res.att, res.att_se))
        except Exception as exc:
            rows.append((est, float("nan"), float("nan"), float("nan"), str(exc)))
    out = args.out / f"ngboost_vs_hgb_dml_{date.today().isoformat()}.md"
    lines = [
        f"# DML nuisance comparison ({date.today().isoformat()})",
        "",
        "| Estimator | ATE | SE | ATT | ATT SE |",
        "|-----------|-----|-----|-----|--------|",
    ]
    for row in rows:
        if len(row) == 5 and isinstance(row[4], str):
            lines.append(f"| {row[0]} | error | — | — | {row[4]} |")
        else:
            lines.append(
                f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} |"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
