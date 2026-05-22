#!/usr/bin/env python3
"""Write calibration gauges for Prometheus textfile collector (optional sidecar)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/validation/calibration_latest.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("observability/prometheus/textfile/calibration.prom"),
    )
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"Report not found: {args.report}")

    data = json.loads(args.report.read_text())
    crps = data.get("crps_1d") or data.get("metrics", {}).get("crps_1d")
    ece = data.get("ece") or data.get("metrics", {}).get("ece")
    lines: list[str] = []
    if crps is not None:
        lines.append("# HELP cocoa_calibration_crps_1d CRPS from latest calibration report")
        lines.append("# TYPE cocoa_calibration_crps_1d gauge")
        lines.append(f"cocoa_calibration_crps_1d {float(crps)}")
    if ece is not None:
        lines.append("# HELP cocoa_calibration_ece Expected calibration error")
        lines.append("# TYPE cocoa_calibration_ece gauge")
        lines.append(f"cocoa_calibration_ece {float(ece)}")
    if not lines:
        raise SystemExit("No crps_1d or ece in report")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
