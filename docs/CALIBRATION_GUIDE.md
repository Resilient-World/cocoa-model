# Calibration guide — probabilistic forecast metrics

This guide explains metrics emitted by `validate_conformal_coverage --calibration-report` and how to remediate failures. See also [conformal_calibration.md](conformal_calibration.md) (online ECI strata) and [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) (spatial-block CV).

## Metrics

| Metric | Meaning | Good value | Remediation |
|--------|---------|------------|-------------|
| **CRPS** | Continuous ranked probability score vs predictive CDF | Lower | Retrain quantile head; refresh ERA5/static features |
| **CRPSS** | `1 − CRPS_model / CRPS_baseline` | > 0 vs climatology | Compare baselines in report; improve signal over persistence |
| **ECE** | Mean \|nominal − empirical\| across quantile levels | < 0.05 | Recalibrate conformal `Q`; adjust quantile spread |
| **PIT χ² p** | Uniformity of probability integral transform | ≥ 0.01 | U-shape → widen intervals; hump → narrow |
| **PIT shape** | `uniform`, `u_shape`, `hump`, `skewed` | `uniform` | Same as PIT p; inspect reliability plot |
| **Sharpness** | Mean prediction interval width | Stable vs release baseline (≤10% regression) | Trade-off with coverage; do not narrow without checking PIT |
| **Energy Score** | Multivariate score (yield × hazard) | Lower | Improve joint ensemble spread in joint head |
| **Coverage** | Fraction of obs inside conformal interval | Within ±2 pp of nominal (90% → 88–92%) | Blocked recalibration (`fit_blocked`, spatial_block) |

## CI gates

`quality_gates.yml` runs:

```bash
python -m models.conformal.validate_conformal_coverage \
  --calibration-gate --calibration-report --synthetic \
  --cv-strategy spatial_block \
  --baseline-calibration tests/fixtures/promotion/baseline_calibration.json
```

Failures:

1. `|empirical_coverage − nominal| > 0.02`
2. `pit_chi2_p < 0.01`
3. `sharpness > 1.10 × baseline_sharpness`

## Commands

```bash
make validate-calibration
make plot-reliability
python -m models.conformal.validate_conformal_coverage --cv-strategy spatial_block --calibration-report
```

## Artifacts

| Path | Content |
|------|---------|
| `reports/validation/calibration_cqr_yield_<date>.json` | Full metrics + per-stratum |
| `reports/validation/calibration_latest.json` | Symlink copy for plotting |
| `reports/validation/reliability_cqr_yield_<date>.png` | Reliability + PIT + CRPSS |

## Libraries

- [properscoring](https://github.com/TheClimateCorporation/properscoring) (Apache-2.0) — CRPS
- [scoringrules](https://github.com/frazane/scoringrules) (MIT) — Energy Score

License boundary: `docs/licensing/PROPERSCORING_SCORINGRULES.md`.
