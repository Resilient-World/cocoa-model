# Sensitivity analysis (Marginal Sensitivity Model)

This project supports two complementary sensitivity tools:

| Method | Module | Question |
|--------|--------|----------|
| Rosenbaum / E-value | `analysis.sensitivity` | How strong must hidden bias be to overturn a **matched-pair** result? |
| **DVDS** (Doubly-Valid / Doubly-Sharp) | `analysis.dvds` | What is the **range of cooperative average treatment effects** if unmeasured confounders are bounded? |

DVDS follows Dorn, Guo & Kallus (2022) under Tan’s **Marginal Sensitivity Model (MSM)**.

## What Λ means (for managers)

- **Λ = 1** — no unmeasured confounding beyond what we already adjust for (standard causal assumption).
- **Λ = 1.5** — we assume unmeasured confounders could change the **odds** of shade-tree adoption by at most **50%** (up or down), within each group of farms that look the same on observed data.
- **Λ = 2** — odds of adoption could at most **double** (or halve) due to unmeasured factors.

Larger Λ widens the bound on the cooperative **average treatment effect (ATE)** on yield (tonnes/ha). If the entire bound interval still lies above zero, the positive intervention story is robust at that level of skepticism.

## API: cooperative bounds on `/simulate-intervention`

Per-farm **avoided loss** and its MC/CQR interval describe the yield **model** for one farm. They are not the same as DVDS bounds.

Optional cooperative MSM bounds (observational panel):

```json
POST /simulate-intervention
{
  "include_sensitivity": true,
  "farm_location": { "lat": 6.5, "lon": -1.2 },
  ...
}
```

Response field `sensitivity_bounds` lists sharp ATE lower/upper bounds and **95% Wald** limits at each Λ in `DVDS_LAMBDA_GRID` (default `1.1, 1.25, 1.5, 2.0`). `tipping_point_lambda` is the smallest Λ in `[1, 10]` where the Wald partial-ID band includes zero.

**Data:** `FARM_PANEL_PARQUET_PATH` (default `data/raw/farm_panel.parquet`). If missing, the API uses a synthetic panel for CI/dev.

## Python usage

```python
from analysis.dvds import dvds_ate, tipping_point

result = dvds_ate(
    snapshot_df,
    treatment_col="received_intervention",
    outcome_col="yield_delta",
    covariate_cols=["soil_quality_index", "historical_rainfall", ...],
    lambda_=1.5,
)
# result.ate_lower, result.ate_upper, result.ate_ci_lower, result.ate_ci_upper

tp = tipping_point(snapshot_df, treatment_col="received_intervention", ...)
```

Validation: `python scripts/validate_dvds.py` → `reports/sensitivity/dvds_validation_<date>.md`.

## References

- Dorn, Guo & Kallus (2022), *Doubly-Valid/Doubly-Sharp Sensitivity Analysis for Causal Inference with Unmeasured Confounding*, arXiv:2112.11449.
- Tan (2006), marginal sensitivity model.
- Zhao, Small & Bhattacharya (2019), bootstrap MSM bounds (comparison in validation script).
