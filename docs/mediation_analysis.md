# Causal mediation analysis (NDE / NIE)

This document describes how the resilient-cocoa-model decomposes **why** an intervention
changes yield along mechanistic paths (microclimate, soil moisture, CSSVD).

## Definitions (VanderWeele 2015)

For binary treatment \(T\) (no intervention vs shade trees), mediator \(M\), and outcome \(Y\) (yield):

- **Total effect (TE):** \(E[Y(1,M(1)) - Y(0,M(0))]\)
- **Natural direct effect (NDE):** \(E[Y(1,M(0)) - Y(0,M(0))]\) — treatment changes outcome holding the mediator at its natural level under control
- **Natural indirect effect (NIE):** \(E[Y(1,M(1)) - Y(1,M(0))]\) — treatment changes outcome only through the mediator
- **Proportion mediated:** \(\mathrm{NIE} / \mathrm{TE}\) (guarded when TE ≈ 0)

Under sequential ignorability and correct nuisance models, TE ≈ NDE + NIE.

## Estimator (Imai, Keele & Yamamoto 2010)

Implementation: `src/analysis/mediation.py` — `mediation_analysis()`.

1. Cross-fit nuisances with HistGradientBoosting (same defaults as `default_nuisance_models` in `heterogeneity.py`):
   - \(e(T \mid X)\)
   - \(e(M \mid X, T)\)
   - \(e(Y \mid X, T, M)\)
2. Per unit, simulate potential outcomes via g-computation:
   - \(M_i(0), M_i(1)\)
   - \(Y_i(0,M_i(0)), Y_i(1,M_i(0)), Y_i(1,M_i(1))\)
3. Bootstrap percentile CIs (default 500 reps offline; API capped via `MEDIATION_N_BOOTSTRAP`, default 200).

## Mediator IDs (intervention API)

| ID | Construct | Source in `/simulate-intervention` |
|----|-----------|-------------------------------------|
| `microclimate` | Composite index from annual mean Δ`tmean`, Δ`vpd`, Δ`rh_mean` | Factual vs counterfactual climate tensors |
| `soil_moisture` | Annual mean `sm_root` (factual − counterfactual) | Same tensors |
| `cssvd_prevalence` | Δ CSSVD loss fraction | `biotic_loss_attribution.projected − baseline` |

Single-farm requests build a **pseudo-panel**: `2 × num_samples` rows (MC draws × treatment arm) with path-level mediator scalars repeated per draw. Bootstrap CIs reflect sampling uncertainty, not a cooperative cohort.

## API usage

```json
POST /simulate-intervention
{
  "farm_location": {"lat": 6.12, "lon": -5.34},
  "farm_size_ha": 3.2,
  "current_yield": 1.8,
  "intervention_type": "shade_trees",
  "decompose_mediators": ["microclimate", "soil_moisture", "cssvd_prevalence"]
}
```

Response field: `mediation.per_mediator[]` with `nde`, `nie`, `total_effect`, CIs, and `rho_critical`.
When multiple mediators are requested, `mediation.path_table` lists ordered path effects.

## ρ sensitivity

Following VanderWeele (2015) §2.4 spirit, adjusted NIE at confounding strength ρ:

\[
\mathrm{NIE}_{\mathrm{adj}}(\rho) = \mathrm{NIE} - \rho \cdot \mathrm{sd}(M) \cdot \mathrm{sd}(Y_{\mathrm{resid}})
\]

`rho_critical` is the smallest ρ in \([0, 0.9]\) where adjusted NIE ≤ 0. The sensitivity curve is returned in offline `MediationResult.sensitivity_curve`.

## Multi-mediator paths

`multi_mediator_decomposition()` reports path-specific effects for an ordered list (demo order: microclimate → soil_moisture → cssvd_prevalence).

## References

- VanderWeele, T. J. (2015). *Explanation in Causal Inference.*
- Imai, K., Keele, L., & Yamamoto, T. (2010). Identification, inference and sensitivity analysis for causal mediation effects.
