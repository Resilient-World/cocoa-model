# Cocoa resilience demo reports (v0.3.0)

This folder holds stakeholder-facing outputs from `scripts/demo_end_to_end.py`.

## Artifacts

| File | Audience | Contents |
|------|----------|----------|
| `e2e_civ_v5.json` | Engineering / integration | Machine-readable results for all eight modules |
| `e2e_civ_v5.md` | Executives / program leads | Plain-language bullets per module |

Run offline (no Earth Engine credentials):

```bash
USE_REAL_FEATURES=false python scripts/demo_end_to_end.py --mock-gee --pretty
```

## Eight modules (plain language)

1. **Forest compliance (EUDR)** — Whisp screening for post-2020 deforestation and protected-area overlap on the farm polygon.
2. **Where cocoa is likely growing (TerraMind+TiM)** — Foundation-model exposure map sample alongside the default ensemble backend.
3. **Historical climate impact (ATTRICI)** — How much yield loss is attributed to observed climate change vs a no-warming counterfactual.
4. **Shade-tree intervention value** — Avoided loss in tonnes and dollars with uncertainty bands (MC or CQR).
5. **Future climate under SSP5-8.5 2050** — CASEJ scenario comparison with optional CorrDiff downscaling when cache and `CORRDIFF_AVAILABLE=true`.
6. **When model confidence drifts (WCTM)** — Drift status from online conformal monitoring on the scenario path.
7. **Hidden bias in cooperative estimates (DVDS) + targeting rules** — Sensitivity bounds at Λ=1.5 and top honest DR policy-tree rules from a synthetic cooperative panel.
8. **Why the intervention works (mediation)** — Natural direct and indirect effects through microclimate, soil moisture, and CSSVD prevalence paths.

## Citations

Source IDs in the JSON `source_attributions` array map to datasets and methods documented in the repository `docs/` folder and `CHANGELOG.md` v0.3.0.
