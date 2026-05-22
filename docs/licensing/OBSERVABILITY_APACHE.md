# OpenTelemetry and Prometheus instrumentation license boundary

| Component | License | Use |
|-----------|---------|-----|
| OpenTelemetry Python SDK and instrumentation | Apache-2.0 | Optional tracing (`src/api/telemetry.py`) |
| prometheus-client | Apache-2.0 | Custom metrics (`src/api/metrics.py`) |
| prometheus-fastapi-instrumentator | Apache-2.0 | HTTP request metrics |

Install via `pip install -e ".[observability]"`. Disabled by default (`OTEL_ENABLED=false`, `PROMETHEUS_ENABLED=false`).
