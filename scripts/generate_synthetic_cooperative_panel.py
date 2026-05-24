#!/usr/bin/env python3
"""Generate a deterministic cooperative panel for causal DAG validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_cooperative_panel(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shade = rng.normal(size=n)
    farm_size = rng.lognormal(mean=1.0, sigma=0.25, size=n)
    rainfall = rng.normal(size=n)
    micro = 0.9 * shade - 0.2 * rainfall + rng.normal(scale=0.25, size=n)
    soil = 0.5 * micro + 0.2 * rainfall + rng.normal(scale=0.35, size=n)
    cssvd = 0.8 * micro + rng.normal(scale=0.25, size=n)
    yield_ = 0.5 * micro - 0.9 * cssvd + rng.normal(scale=0.35, size=n)
    return pd.DataFrame(
        {
            "shade_trees": shade,
            "farm_size_ha": farm_size,
            "historical_rainfall": rainfall,
            "microclimate_index": micro,
            "soil_moisture_delta": soil,
            "cssvd_prevalence_delta": cssvd,
            "yield": yield_,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/sample_cooperative_panel.csv")
    )
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic_cooperative_panel(args.rows, args.seed).to_csv(args.output, index=False)
    sys.stdout.write(f"{args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
