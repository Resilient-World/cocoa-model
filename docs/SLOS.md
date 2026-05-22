# Service level objectives (Cocoa Model API)

Operational targets for the FastAPI inference service. Metrics and alerts assume
`PROMETHEUS_ENABLED=true` and the dashboards in `grafana/dashboards/`.

See also [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) and [conformal_calibration.md](conformal_calibration.md).

## SLIs and SLOs

| Endpoint / signal | SLI | SLO target |
|-------------------|-----|------------|
| `POST /simulate-intervention` | p95 latency | < 500 ms |
| | Error rate (5xx + mapped failures) | < 0.5% |
| | Availability | 99.5% monthly |
| `POST /simulate-scenario` | p95 latency | < 1500 ms |
| | Availability | 99.0% monthly |
| Conformal coverage | \|empirical − nominal\| over 1000-sample window | < 3 pp |
| Drift (WCTM) | `exp(log_martingale) > 100` | Page on-call |

Prometheus series: `cocoa_inference_latency_seconds`, `cocoa_simulation_errors_total`,
`cocoa_conformal_coverage_running_avg`, `cocoa_drift_score`.

## Error budgets

- **99.5% availability** → ~3.6 hours downtime per 30-day month.
- **99.0% availability** → ~7.2 hours downtime per 30-day month.

Burn when `error_rate > (1 - SLO)` for sustained windows.

## Example PromQL (burn-rate style)

**simulate-intervention error rate (5m critical):**

```promql
sum(rate(cocoa_simulation_errors_total{endpoint="/simulate-intervention"}[5m]))
/
sum(rate(http_requests_total{handler="/simulate-intervention"}[5m]))
> 0.005
```

**p95 latency (5m):**

```promql
histogram_quantile(
  0.95,
  sum(rate(cocoa_inference_latency_seconds_bucket{endpoint="/simulate-intervention"}[5m])) by (le)
) > 0.5
```

Reference alert rules: `observability/prometheus/alerts.yml`.

## Load testing

k6 smoke/nightly thresholds mirror intervention p95 (`p(95)<500`). See `tests/loadtest/k6/`
and `make loadtest URL=http://localhost:8000`.
