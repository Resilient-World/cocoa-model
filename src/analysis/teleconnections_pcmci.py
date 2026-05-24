"""PCMCI+ teleconnection discovery through a Tigramite subprocess boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

TELECONNECTION_COLUMNS = (
    "nino34",
    "atl3",
    "iod_dmi",
    "cocoa_precipitation",
    "cocoa_vpd",
    "yield_anomaly",
)


def _write_plot(edges: list[dict[str, Any]], output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * max(1, len(edges)))))
    labels = [
        f"{edge['source']} → {edge['target']} (lag {edge['lag']})"
        for edge in sorted(edges, key=lambda e: float(e.get("strength", 0.0)), reverse=True)
    ]
    strengths = [
        float(edge.get("strength", 0.0))
        for edge in sorted(edges, key=lambda e: float(e.get("strength", 0.0)), reverse=True)
    ]
    if strengths:
        ax.barh(labels, strengths)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No PCMCI+ links discovered", ha="center", va="center")
        ax.set_axis_off()
    ax.set_xlabel("Link strength")
    ax.set_title("PCMCI+ teleconnection DAG")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def discover_teleconnection_pcmci(
    panel: pd.DataFrame,
    *,
    output_json: Path = Path("reports/causal/teleconnections_pcmci.json"),
    output_png: Path = Path("reports/causal/teleconnections_pcmci.png"),
    max_lag_months: int = 12,
    alpha: float = 0.05,
    shim_path: Path | None = None,
) -> dict[str, Any]:
    """Run Tigramite PCMCI+ on climate-index and cocoa-belt anomaly columns."""
    missing = [col for col in TELECONNECTION_COLUMNS if col not in panel.columns]
    if missing:
        raise ValueError(f"teleconnection panel missing required columns: {missing}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    work_csv = output_json.with_suffix(".input.csv")
    panel.loc[:, TELECONNECTION_COLUMNS].to_csv(work_csv, index=False)
    script = shim_path or Path(__file__).resolve().parents[2] / "scripts" / "tigramite_cli_shim.py"
    cmd = [
        sys.executable,
        str(script),
        "--input-csv",
        str(work_csv),
        "--output-json",
        str(output_json),
        "--max-lag",
        str(max_lag_months),
        "--alpha",
        str(alpha),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    edges = list(payload.get("edges", []))
    _write_plot(edges, output_png)
    payload["plot_path"] = str(output_png)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cast(dict[str, Any], payload)
