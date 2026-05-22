#!/usr/bin/env python3
"""Generate mpdta-like fixture and CS-DID benchmark JSON for tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from analysis.csdid import CallawaySantAnna

OUT_DIR = _REPO / "tests" / "fixtures" / "csdid"


def build_mpdta_like(*, seed: int = 42) -> pd.DataFrame:
    """
    County teen-employment panel mimicking ``did::mpdta`` structure.

    500 counties, 2004-2007, cohorts 2004 / 2006 / 2007, never-treated share ~40%.
    """
    rng = np.random.default_rng(seed)
    n_county = 500
    years = [2004, 2005, 2006, 2007]
    counties = np.arange(1, n_county + 1)
    cohort_draw = rng.choice([0, 2004, 2006, 2007], size=n_county, p=[0.4, 0.2, 0.2, 0.2])
    rows: list[dict] = []
    for c, g in zip(counties, cohort_draw):
        fe = rng.normal(0, 0.3)
        for yr in years:
            lpop = rng.normal(10, 0.5)
            treat = int(g > 0 and yr >= g)
            effect = -0.02 * treat if g > 0 else 0.0
            lemp = fe + 0.01 * (yr - 2004) + effect + rng.normal(0, 0.05)
            rows.append(
                {
                    "countyreal": int(c),
                    "year": yr,
                    "first.treat": float(g),
                    "lpop": lpop,
                    "lemp": lemp,
                    "treat": treat,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[df["first.treat"] == 0, "first.treat"] = np.nan
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_mpdta_like()
    csv_path = OUT_DIR / "mpdta.csv"
    df.to_csv(csv_path, index=False)

    est = CallawaySantAnna(
        df,
        unit_col="countyreal",
        time_col="year",
        treat_time_col="first.treat",
        outcome_col="lemp",
        covariate_cols=["lpop"],
        n_boot=199,
        random_state=42,
    )
    benchmarks: dict[str, float] = {"simple_att": est.simple_att().att}
    for g in [2004, 2006, 2007]:
        for t in [2004, 2005, 2006, 2007]:
            if t >= g:
                r = est.att_gt(g, t)
                benchmarks[f"att_gt_{g}_{t}"] = r.att

    bench_path = OUT_DIR / "mpdta_benchmarks.json"
    with bench_path.open("w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Wrote {csv_path} ({len(df)} rows) and {bench_path}")


if __name__ == "__main__":
    main()
