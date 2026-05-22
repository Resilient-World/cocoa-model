# Incident runbook

Quick recipes for on-call. Cross-links: [SLOS.md](SLOS.md), Grafana dashboards,
Jaeger (`http://localhost:16686` when using `docker-compose.observability.yml`).

## 1. Drift spike (WCTM)

**Symptoms:** `cocoa_drift_score` > 100, drift alarms in API responses.

**PromQL:**

```promql
cocoa_drift_score{model="wctm"}
```

**Logs:** `structlog` fields `trace_id`, stratum key `scenario:horizon:region`.

**Actions:**

1. Inspect stratum in `/drift-status/{stratum}`.
2. Widen conformal (`DRIFT_INFLATION_FACTOR`) or pause promotion.
3. Retrain / refresh CQR if `concept_shift` diagnosis persists.

## 2. Coverage drop

**Symptoms:** `cocoa_conformal_coverage_running_avg` deviates >3pp from 0.9.

**PromQL:**

```promql
abs(cocoa_conformal_coverage_running_avg - 0.9) > 0.03
```

**Actions:**

1. Run `make validate-calibration` and review `reports/validation/calibration_latest.json`.
2. Check online store `fit_blocked` / stratum counts.
3. Re-fit conformal per [conformal_calibration.md](conformal_calibration.md).

## 3. Latency regression

**Symptoms:** SLO burn on `cocoa_inference_latency_seconds` p95.

**Jaeger:** Filter service `cocoa-model-api`; compare spans
`feature_resolver.resolve_climate`, `casej_surrogate.forward`, `scenario_builder.build`.

**Actions:**

1. Confirm `USE_REAL_FEATURES` vs geo_mock in staging.
2. Warm feature cache / disk cache.
3. Scale replicas if CPU-saturated.

## 4. Mediation NaN

**Symptoms:** Missing `mediation` block or invalid ratios in logs.

**Logs:** event path `mediation.decompose`; check bootstrap count `MEDIATION_N_BOOTSTRAP`.

**Actions:**

1. Reduce mediators requested in `decompose_mediators`.
2. Increase `n_bootstrap` only if latency budget allows.
3. Verify panel / tensor finite values.

## 5. EUDR timeout

**Symptoms:** `cocoa_eudr_compliance_total{status="timeout"}` increasing.

**PromQL:**

```promql
sum(rate(cocoa_eudr_compliance_total{status="timeout"}[5m]))
```

**Actions:**

1. Verify `WHISP_API_KEY` and mock vs live Whisp.
2. Disable GEE screening on hot path (`use_gee_fdp_screening=false`).
3. Call `/eudr-due-diligence` async/offline for batch plots.
