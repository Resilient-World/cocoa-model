# DVDS validation report

> **Status:** Smoke run only (`--reps 5`, `n=500`). The full production gate (`python scripts/validate_dvds.py --reps 200 --n 1000`) was **deferred** until more compute is available—see [`docs/sensitivity.md`](../../docs/sensitivity.md) and [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md). Do not use this file as the signed-off validation artifact.

Generated: 2026-05-21

## Section 7.1 binary DGP (n=1000)

Monte Carlo true ATE: **0.2141** (5 replications)

| Λ | Point bound coverage | Wald CI coverage | DVDS width ≤ Zhao bootstrap |
|---|---------------------|------------------|----------------------------|
| 1.0 | 0.0% | 60.0% | 100.0% |
| 1.5 | 20.0% | 100.0% | 100.0% |
| 2.0 | 0.0% | 100.0% | 100.0% |
| 2.5 | 0.0% | 100.0% | 100.0% |

## Farm panel synthetic ATT

True ATT (tonnes/ha): **0.35**

| Λ | Interval width | Contains true ATT |
|---|----------------|-------------------|
| 1.1 | 0.0981 | no |
| 1.5 | 0.4188 | yes |
| 2.0 | 0.7696 | yes |
| 3.0 | 1.4717 | yes |
