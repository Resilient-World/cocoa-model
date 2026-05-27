# TSFM Compute Requirements

Time-series foundation models (TSFMs) for cocoa yield forecasting.

## Model Sizes

| Model | HF Repo | Params | Disk (approx) | License |
|-------|---------|--------|---------------|---------|
| Time-MoE-50M | `Maple728/TimeMoE-50M` | 50M | ~200 MB | Apache-2.0 |
| Chronos-2 | `amazon/chronos-2` | ~200M | ~800 MB | Apache-2.0 |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 200M | ~800 MB | Apache-2.0 |
| Moirai-2-R-small | `Salesforce/moirai-2.0-R-small` | ~30M | ~120 MB | Apache-2.0 |

All models are downloaded from Hugging Face Hub on first use (cached in `~/.cache/huggingface/`).

## Inference Latency

Approximate single-forecast latency (24-month history → 12-month horizon, 100 samples):

| Model | CPU (M1 Pro) | GPU (A10G) |
|-------|-------------|------------|
| Time-MoE-50M | ~2–5 s | ~0.5 s |
| Chronos-2 | ~10–20 s | ~1–2 s |
| TimesFM 2.5 | ~8–15 s | ~1–2 s |
| Moirai-2-R-small | ~3–8 s | ~0.5 s |

**Ensemble (all 4, parallel):** ~10–20 s CPU, ~2–4 s GPU.

## Recommendations

- **CPU-only deployments:** Use Time-MoE-50M as the primary model. It achieves competitive accuracy
  (54.9% MAE improvement over USDA baseline on wheat per Wang & Zhang 2026) with manageable
  CPU latency (~2–5 s per forecast).
- **GPU deployments:** Run the full ensemble for maximum accuracy. The 4-model ensemble
  completes in ~2–4 s on an A10G.
- **Batch forecasting:** For bulk backtesting or scenario sweeps, batch multiple forecasts
  to amortize model loading overhead.

## Environment Variables

```bash
TSFM_ENABLED=true           # Enable TSFM ensemble (default: false)
TSFM_PRIMARY=timemoe        # Primary model for best mode (chronos-2|timesfm|timemoe|moirai)
TSFM_ENSEMBLE_MODE=nnls     # Aggregation: mean|nnls|best
TSFM_WEIGHTS_PATH=config/tsfm_weights.yaml
```

## Dependencies

Install TSFM-specific dependencies:

```bash
# Time-MoE (recommended default)
pip install transformers torch

# Chronos-2
pip install chronos

# TimesFM 2.5
pip install timesfm

# Moirai-2
pip install uni2ts

# Ensemble weight fitting
pip install scipy  # for scipy.optimize.nnls
```

## References

- Wang & Zhang (2026). "Benchmarking Time Series Foundation Models for Agricultural
  Commodity Price Forecasting." arXiv:2601.06371.
- Time-MoE: Shi et al. (2024). "Time-MoE: Billion-Scale Time Series Foundation Models
  with Mixture of Experts." arXiv:2409.16040.
- Chronos: Ansari et al. (2024). "Chronos: Learning the Language of Time Series."
  arXiv:2403.07815.
- TimesFM: Das et al. (2024). "A Decoder-only Foundation Model for Time-Series
  Forecasting." arXiv:2310.10688.
- Moirai: Woo et al. (2024). "Unified Training of Universal Time Series Forecasting
  Transformers." arXiv:2402.02592.
