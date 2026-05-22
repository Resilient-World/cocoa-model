#!/usr/bin/env python3
"""
Learn DR policy targeting rules from a farm panel CSV and write a markdown report.

Compares tree/forest policy value to a greedy CATE/cost baseline when --budget is set.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from analysis.policy_targeting import (
    learn_policy_forest,
    learn_policy_tree,
    render_policy_rules,
    render_policy_rules_from_forest,
)

REPORT_DIR = _REPO / "reports" / "targeting"


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn DR policy rules from a panel CSV")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--covariates", required=True, help="Comma-separated column names")
    parser.add_argument("--cost-col", default=None)
    parser.add_argument("--region", default="all", help="Slug for output filename")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown path (default reports/targeting/policy_rules_{region}_{date}.md)",
    )
    parser.add_argument("--learner", choices=("tree", "forest"), default="tree")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--intervention-cost", type=float, default=0.0)
    parser.add_argument("--treatment-label", default="treat_with_shade_trees")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--refit-bootstrap", action="store_true", help="Reserved; fixed-policy bootstrap only"
    )
    args = parser.parse_args()
    del args.refit_bootstrap

    df = pd.read_csv(args.panel)
    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]
    report_date = date.today().isoformat()
    out_path = args.out or REPORT_DIR / f"policy_rules_{args.region}_{report_date}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        treatment_col=args.treatment,
        outcome_col=args.outcome,
        covariate_cols=covariates,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        cost_col=args.cost_col,
        n_folds=args.n_folds,
        random_state=args.random_state,
        n_bootstrap=args.n_bootstrap,
        intervention_cost_usd_per_farm=args.intervention_cost,
        budget=args.budget,
        recommended_treatment_label=args.treatment_label,
    )

    if args.learner == "forest":
        result = learn_policy_forest(**common, n_estimators=args.n_estimators)
        rules = render_policy_rules_from_forest(
            result, recommended_treatment_label=args.treatment_label
        )
        method = "policy_forest"
    else:
        result = learn_policy_tree(**common)
        rules = render_policy_rules(result, recommended_treatment_label=args.treatment_label)
        method = "policy_tree"

    lines = [
        f"# Policy targeting rules ({args.region}, {report_date})",
        "",
        f"- **Method:** `{method}` (honest DR, EconML)",
        f"- **Samples:** {len(df)}",
        f"- **Cost-aware:** {result.cost_aware}",
        "",
        "## Policy value",
        "",
        f"- DR policy value: **{result.policy_value_estimate:.4f}** "
        f"(95% CI {result.policy_value_ci[0]:.4f} – {result.policy_value_ci[1]:.4f})",
    ]
    if result.greedy_policy_value is not None:
        lines.append(f"- Greedy CATE/cost baseline: **{result.greedy_policy_value:.4f}**")
    lines.extend(["", "## Rendered rules", ""])
    for i, rule in enumerate(rules, start=1):
        lines.append(f"{i}. {rule}")
    lines.extend(["", "## Leaf summary", ""])
    if not result.leaf_summary.empty:
        lines.append(result.leaf_summary.to_markdown(index=False))
    else:
        lines.append("_No leaf summary rows._")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
