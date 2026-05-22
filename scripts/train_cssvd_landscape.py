#!/usr/bin/env python3
"""
Train LandscapeCSSVDModel on Dumont et al. supplement + landscape covariates.

Example::

    python scripts/train_cssvd_landscape.py --synthetic --checkpoint models/cssvd_landscape.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.dumont_supplement import (
    DEFAULT_SUPPLEMENT_PATH,
    DEFAULT_SYNTHETIC_PATH,
    generate_synthetic_supplement,
    join_exposure_features,
    load_dumont_plots,
    normalize_dumont_columns,
)
from hazards.cssvd_landscape import (
    DEFAULT_CHECKPOINT,
    HORIZON_MONTHS,
    NUMERIC_FEATURES,
    STRAIN_PREFIX,
    LandscapeCSSVDModel,
    fit_synthetic_demo,
    incidence_probability_at_horizon,
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    strain_cols = [f"{STRAIN_PREFIX}{r}" for r in ("1A", "1B", "1C")]
    return list(NUMERIC_FEATURES) + [c for c in strain_cols if c in df.columns]


def _blocked_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "country" in df.columns:
        countries = df["country"].astype(str).unique()
        if len(countries) >= 2:
            holdout = countries[-1]
            test = df[df["country"].astype(str) == holdout]
            train = df[df["country"].astype(str) != holdout]
            if len(train) > 0 and len(test) > 0:
                return train, test
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSSVD landscape survival model")
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT_PATH)
    parser.add_argument(
        "--synthetic", action="store_true", help="Use/generate synthetic supplement"
    )
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--use-gee", action="store_true", help="Sample landscape features via GEE")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--reports", type=Path, default=_REPO_ROOT / "reports" / "cssvd_landscape_metrics.json"
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo-only", action="store_true", help="Fit internal synthetic demo (no CSV)"
    )
    args = parser.parse_args()

    if args.demo_only:
        model = fit_synthetic_demo(random_state=args.seed)
        model.save(args.checkpoint)
        print(f"Saved demo model to {args.checkpoint}")
        return

    if args.synthetic:
        generate_synthetic_supplement(
            args.n_synthetic,
            seed=args.seed,
            output_path=DEFAULT_SYNTHETIC_PATH,
        )
        plots = normalize_dumont_columns(pd.read_csv(DEFAULT_SYNTHETIC_PATH))
    else:
        plots = load_dumont_plots(args.supplement)

    merged = join_exposure_features(
        plots,
        args.year,
        use_gee=args.use_gee,
        refresh_cache=args.synthetic,
    )
    feat_cols = _feature_columns(merged)
    train_df, val_df = _blocked_split(merged, test_size=0.2, seed=args.seed)

    X_train = train_df[feat_cols]
    X_val = val_df[feat_cols]
    y_train_dur = train_df["duration"].to_numpy(dtype=np.float64)
    y_train_ev = train_df["event"].to_numpy(dtype=np.int32)
    y_val_dur = val_df["duration"].to_numpy(dtype=np.float64)
    y_val_ev = val_df["event"].to_numpy(dtype=np.int32)

    model = LandscapeCSSVDModel(
        n_estimators=args.n_estimators,
        n_bootstrap=args.n_bootstrap,
        random_state=args.seed,
    )
    train_metrics = model.fit(X_train, y_train_dur, y_train_ev, fit_bootstrap=True)
    model.save(args.checkpoint)

    val_probs = incidence_probability_at_horizon(model._model, X_val, horizon_months=HORIZON_MONTHS)  # type: ignore[arg-type]
    brier = float(np.mean((val_probs - y_val_ev) ** 2))

    metrics = {
        **train_metrics,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "brier_12mo_val": brier,
        "horizon_months": HORIZON_MONTHS,
        "checkpoint": str(args.checkpoint),
    }
    args.reports.parent.mkdir(parents=True, exist_ok=True)
    args.reports.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
