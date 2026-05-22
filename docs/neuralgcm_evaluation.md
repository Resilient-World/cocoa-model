# NeuralGCM and ACE2-ERA5 evaluation

## When to use

| Horizon | Recommended backend |
|---------|---------------------|
| 1–15 years regional | `neuralgcm` (opt-in, `NEURALGCM_ENABLED=true`) |
| SSP 2030–2080 with km-scale structure | `corrdiff` or `linear_delta` (default) |
| Beyond 2050 multi-decadal | `corrdiff` or `linear_delta` — **not** NeuralGCM |

## Documented limitations

NeuralGCM does **not** reproduce:

- Quasi-biennial oscillation (QBO, ~28 months)
- Propagating Southern Annular Mode (SAM, ~150 days)

See Baxter et al. (2025) and `reports/scenario/neuralgcm_vs_corrdiff_<date>.md`.

## Validation

```bash
python scripts/validate_neuralgcm_scenario.py
```

Full CRPS backtest against observed ERA5 at cocoa pixels (Ghana, CIV, Cameroon) requires GPU and ERA5 Zarr paths.
