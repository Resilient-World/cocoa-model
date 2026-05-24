#!/usr/bin/env python3
"""Validate the assumed mediation DAG against discovered causal structure."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from analysis.causal_discovery import (
    ASSUMED_MEDIATION_EDGES,
    compare_with_assumed_dag,
    discover_dag_ges,
    discover_dag_notears,
    discover_dag_pc,
    edges_to_jsonable,
    ensemble_discovered_dag,
)
from scripts.generate_synthetic_cooperative_panel import synthetic_cooperative_panel


def _load_panel(path: Path, synthetic: bool) -> pd.DataFrame:
    if path.is_file() and not synthetic:
        return pd.read_csv(path)
    return synthetic_cooperative_panel()


def _format_edges(edges: object) -> str:
    if isinstance(edges, dict):
        pairs = [f"{src} → {dst} ({conf:.2f})" for (src, dst), conf in edges.items()]
    else:
        pairs = [f"{src} → {dst}" for src, dst in edges]
    return "\n".join(f"- {pair}" for pair in pairs) or "- none"


def write_report(
    *,
    output_dir: Path,
    pc: dict[tuple[str, str], float],
    notears: dict[tuple[str, str], float],
    ges: dict[tuple[str, str], float],
    ensemble: dict[tuple[str, str], float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "discovered_dag_latest.json"
    latest_path.write_text(
        json.dumps({"edges": edges_to_jsonable(ensemble)}, indent=2),
        encoding="utf-8",
    )
    comparison = compare_with_assumed_dag(ensemble)
    recommendation = "ASSUMED DAG VALIDATED"
    if comparison.edges_only_discovered or comparison.edges_only_assumed:
        discovered = ", ".join(f"{a}->{b}" for a, b in comparison.edges_only_discovered) or "none"
        assumed = ", ".join(f"{a}->{b}" for a, b in comparison.edges_only_assumed) or "none"
        recommendation = (
            "REVIEW REQUIRED "
            f"(discovered edges {discovered} not in assumed; assumed edges {assumed} not discovered)"
        )
    report_path = output_dir / f"dag_validation_{date.today().isoformat()}.md"
    report_path.write_text(
        "\n".join(
            [
                "# Causal DAG validation",
                "",
                "## Assumed DAG",
                _format_edges(ASSUMED_MEDIATION_EDGES),
                "",
                "## PC discovered DAG",
                _format_edges(pc),
                "",
                "## NOTEARS-MLP discovered DAG",
                _format_edges(notears),
                "",
                "## GES discovered DAG",
                _format_edges(ges),
                "",
                "## Ensemble confidence DAG",
                _format_edges(ensemble),
                "",
                "## DAGComparisonReport",
                "```json",
                json.dumps(comparison.to_dict(), indent=2),
                "```",
                "",
                "## Recommendation",
                recommendation,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "sample_cooperative_panel.csv",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "causal")
    args = parser.parse_args(argv)
    panel = _load_panel(args.panel, args.synthetic)
    dag_cols = [
        "shade_trees",
        "microclimate_index",
        "cssvd_prevalence_delta",
        "yield",
    ]
    data = panel.loc[:, [col for col in dag_cols if col in panel.columns]]
    pc = discover_dag_pc(data)
    notears = discover_dag_notears(data)
    ges = discover_dag_ges(data)
    ensemble = ensemble_discovered_dag(data)
    report = write_report(
        output_dir=args.output_dir,
        pc=pc,
        notears=notears,
        ges=ges,
        ensemble=ensemble,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
