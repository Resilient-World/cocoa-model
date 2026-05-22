# TCAV interpretability

## Endpoint

`POST /interpret` (requires `INTERPRET_ENABLED=true` and optional `INTERPRET_AUTH_TOKEN`).

Returns per-concept TCAV scores for a farm location and intervention type.

## Concepts

| ID | Meaning |
|----|---------|
| `drought_year` | Below-median growing-season precipitation |
| `shade_present` | Elevated shade fraction in static features |
| `high_rainfall` | Upper-quartile precipitation |
| `cssvd_zone` | High soil-clay / CSSVD-risk proxy |
| `mid_crop_season` | Mid-season temperature pattern |

## Reading scores

- **Score** — fraction of concept examples where the gradient aligns with the concept direction (0–1).
- **p_value** — two-sided t-test vs random concept baselines; values below 0.05 suggest the concept is statistically distinguishable from noise.

Regulator-facing wording: a high score with low p-value indicates the yield model’s internal representation is sensitive to that concept when recommending the intervention.

## Offline plots

```bash
python scripts/run_tcav_analysis.py
```

Outputs under `reports/tcav/`.
