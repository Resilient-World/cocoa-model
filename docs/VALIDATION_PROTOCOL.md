# Validation protocol — spatial and temporal cross-validation

This document defines how we evaluate geospatial and time-series models in **resilient-cocoa-model**, following Roberts et al. (2017), *Ecography*, [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).

## When to use each CV strategy

| Strategy | Use when | Leakage risk if misused |
|----------|----------|-------------------------|
| **Random** | Debugging, fast smoke tests | **High** under spatial autocorrelation — optimistic mIoU/F1/coverage |
| **Spatial block** | Segmentation, yield, conformal calibration (production) | Low — held-out blocks separated by ≥ block size |
| **Buffered LOO** | Fine-grained spatial diagnostics, small n | Medium — buffer must exceed variogram range |
| **Forward chain** | National/annual panels, scenario backtests | High if future years leak into training |
| **Season-aware** | Phenology-sensitive daily features | Seasonal non-stationarity |

**Production claims** (insurer/regulator-facing) must use **spatial-block** metrics at documented block size (default 50 km Ghana/CIV). Random CV may be reported only as a **secondary diagnostic**.

## Choosing block size (Roberts Step 1)

1. Fit a baseline model and compute residuals on a georeferenced validation set.
2. Run `compute_residual_variogram` in [`src/validation/spatial_cv.py`](../src/validation/spatial_cv.py).
3. Set `block_size_km = max(1.5 × range_km, floor_km)` via `recommend_block_size_km`.

```bash
python scripts/validate_spatial_holdout.py --region ghana --block-size-km 50
```

Variogram diagnostics are written to `reports/validation/figures/variogram_<date>.png`.

## Conformal coverage

- Train CQR with blocked calibration:

```bash
python scripts/train_cqr.py --synthetic --cv-strategy spatial_block --block-size-km 50
```

- Evaluate all strategies:

```bash
python -m models.conformal.validate_conformal_coverage --cv-strategy all --out reports/conformal
```

**Gate:** spatial-block empirical coverage ∈ [88%, 92%] for 90% nominal intervals.

## Temporal holdout (2018–2024)

```bash
python scripts/validate_temporal_holdout.py
make validate-temporal
```

Reports CRPS proxy, reliability, and PIT histograms under `reports/validation/temporal_cv_<date>.md`.

## Makefile shortcuts

```bash
make validate-spatial REGION=ghana
make validate-temporal
```

## Data contracts (Pandera)

Ingestion entrypoints validate outputs via [`src/data/schemas.py`](../src/data/schemas.py). Failures raise `ValueError` with the first schema violation cases.

## Nightly CI

`.github/workflows/quality_gates.yml` runs spatial and temporal validation scripts on a schedule (`--quick`) and uploads reports as artifacts.

## References

- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40:913–929.
- Le Rest, K., et al. (2014). Spatial leave-one-out cross-validation for variable selection in remote sensing.
