# Causal discovery for DAG validation

This layer validates the assumed shade-trees → microclimate → CSSVD → yield mediation DAG and discovers lag-aware teleconnection links for the teleconnection GNN.

## Methods

| Method | Use | Output |
|--------|-----|--------|
| PC | Fast constraint-based skeleton checks when variables are few and roughly Gaussian | Directed/partially directed edge set |
| NOTEARS-MLP | Non-linear additive-noise discovery when functional form matters | Weighted DAG |
| GES | Score-based search for higher-dimensional cooperative panels | Candidate DAG under BIC-style score |
| PCMCI+ | Monthly time-series teleconnections with lag 0–12 months | Lag-aware DAG over climate indices and cocoa-belt anomalies |

`analysis.causal_discovery.ensemble_discovered_dag()` returns edge confidence as the fraction of PC, NOTEARS, and GES that include each edge. `scripts/validate_causal_dag.py` writes the latest ensemble to `reports/causal/discovered_dag_latest.json`.

## Running validation

```bash
PYTHONPATH=src python scripts/validate_causal_dag.py --synthetic
```

The report is written to `reports/causal/dag_validation_<date>.md` and includes:

- assumed mediation DAG,
- PC, NOTEARS-MLP, and GES discovered DAGs,
- ensemble confidence DAG,
- `DAGComparisonReport`,
- recommendation: `ASSUMED DAG VALIDATED` or `REVIEW REQUIRED`.

## Mediation opt-in

The API keeps the hard-coded mediation chain by default. To use the discovered mediator order:

```bash
MEDIATION_USE_DISCOVERED_DAG=true
```

When enabled, `api.mediation.compute_intervention_mediation()` reads `reports/causal/discovered_dag_latest.json`, orders requested mediators from that DAG where possible, and returns `mediation.dag_source = "discovered"`. Otherwise responses return `dag_source = "assumed"`.

## Teleconnection PCMCI+

Use `analysis.teleconnections_pcmci.discover_teleconnection_pcmci()` with a monthly panel containing:

- `nino34`,
- `atl3`,
- `iod_dmi`,
- `cocoa_precipitation`,
- `cocoa_vpd`,
- `yield_anomaly`.

Tigramite is GPL-3, so the project calls it only through `scripts/tigramite_cli_shim.py`, matching the ATTRICI subprocess-boundary pattern. Outputs are written to `reports/causal/teleconnections_pcmci.json` and `reports/causal/teleconnections_pcmci.png`.

## Limitations

- **Causal sufficiency:** PC, NOTEARS, and GES assume relevant common causes are measured. Hidden management quality, local disease pressure, or market shocks can create spurious edges.
- **Faithfulness:** Conditional independences in the data must reflect the true graph. Near-canceling biological pathways can hide real causal links.
- **Sample size:** Spirtes-Glymour-style guidance implies sample size must grow quickly with graph degree and conditioning-set size. Treat small panels (<200 rows) as screening only; prefer 500+ rows for the four-node mediation DAG and more for high-dimensional cooperative panels.
- **Directionality:** Observational discovery validates consistency with the assumed DAG; it does not replace agronomic identification or intervention evidence.

## Policy-tree feature engineering

Validated mediator edges identify stable path features for policy targeting. If the ensemble repeatedly supports `microclimate_index → cssvd_prevalence_delta → yield`, include microclimate and CSSVD deltas as policy-tree covariates or rule constraints. Teleconnection PCMCI+ links can similarly define lagged ENSO/ATL3/IOD features before running the policy-tree workflow in `docs/policy_targeting.md`.
