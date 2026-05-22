# NeuralGCM vs CorrDiff skill (2026-05-22)

## CRPS at cocoa pixels (Ghana, CIV, Cameroon)

| Backend | CRPS (stub) | Notes |
|---------|-------------|-------|
| neuralgcm_stub | 0.42 | Run full backtest on GPU with ERA5 Zarr |
| corrdiff_cache | 0.38 | Precomputed Zarr ensemble |
| linear_delta | 0.45 | Default production path |

## Limitations (Baxter et al. 2025)

- NeuralGCM does **not** capture QBO (~28 month) or propagating SAM (~150 day) variability.
- Recommended for **1–15 year** regional tropospheric downscaling where synoptic dynamics dominate.
- For SSP horizons beyond **2050**, use **CorrDiff-CMIP6** or **linear_delta** (see docs/neuralgcm_evaluation.md).
