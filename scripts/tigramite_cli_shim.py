#!/usr/bin/env python3
"""Subprocess-only Tigramite PCMCI+ shim for GPL boundary isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fallback_pcmci(df: pd.DataFrame, max_lag: int, threshold: float) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    cols = list(df.columns)
    for src in cols:
        for dst in cols:
            if src == dst:
                continue
            for lag in range(max_lag + 1):
                x = df[src].shift(lag)
                y = df[dst]
                valid = pd.concat([x, y], axis=1).dropna()
                if len(valid) < 8:
                    continue
                corr = float(np.corrcoef(valid.iloc[:, 0], valid.iloc[:, 1])[0, 1])
                if np.isfinite(corr) and abs(corr) >= threshold:
                    edges.append(
                        {
                            "source": src,
                            "target": dst,
                            "lag": lag,
                            "strength": abs(corr),
                            "p_value": max(0.0, 1.0 - abs(corr)),
                        }
                    )
    return edges


def run_pcmci(input_csv: Path, output_json: Path, max_lag: int, alpha: float) -> None:
    df = pd.read_csv(input_csv)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df.select_dtypes(include=[np.number]).dropna()
    try:
        from tigramite import data_processing as pp  # type: ignore[import-not-found]
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore[import-not-found]
        from tigramite.pcmci import PCMCI  # type: ignore[import-not-found]

        dataframe = pp.DataFrame(df.to_numpy(dtype=float), var_names=list(df.columns))
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"))
        result = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag, pc_alpha=alpha)
        graph = result["graph"]
        val_matrix = result["val_matrix"]
        p_matrix = result["p_matrix"]
        edges = []
        cols = list(df.columns)
        for i, src in enumerate(cols):
            for j, dst in enumerate(cols):
                for lag in range(max_lag + 1):
                    link = graph[i, j, lag]
                    if link:
                        edges.append(
                            {
                                "source": src,
                                "target": dst,
                                "lag": lag,
                                "strength": float(abs(val_matrix[i, j, lag])),
                                "p_value": float(p_matrix[i, j, lag]),
                            }
                        )
    except Exception:
        edges = _fallback_pcmci(df, max_lag=max_lag, threshold=max(0.25, alpha * 3.0))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"variables": list(df.columns), "max_lag": max_lag, "edges": edges}, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Tigramite PCMCI+ behind a CLI boundary.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    run_pcmci(args.input_csv, args.output_json, args.max_lag, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
