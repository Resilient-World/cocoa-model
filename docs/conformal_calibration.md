# Online conformal calibration for `/simulate-scenario`

Production scenario simulations use **ECI-Integral** online conformal calibration by default (`CONFORMAL_METHOD=eci_integral`). Static split-CQR remains available via `CONFORMAL_METHOD=split_cqr`.

## Stratum keys

State is keyed by `{scenario}:{horizon_year}:{region}` — 48 strata (8 FDP regions × 2 SSPs × 3 horizons).

Examples:

- `ssp245:2030:ghana`
- `ssp585:2080:colombia`

## Persistence

| File | Purpose |
|------|---------|
| `data/processed/conformal_initial_state.json` | Bootstrap `q_t` per stratum (from calibration script) |

**CorrDiff downscaling:** When `POST /simulate-scenario` uses `downscaling_method=corrdiff`, online conformal and drift keys append `:corrdiff` (e.g. `ssp245:2050:ghana:corrdiff`). Linear delta-change traffic keeps the original 48 keys without the suffix.
| `data/processed/online_conformal_state.json` | Live `q_t` + rolling coverage window (last 1000 calls) |

Set `REDIS_URL` to mirror the JSON blob in Redis (`online_conformal_state` key). If Redis is unavailable, the API falls back to the JSON file.

## Bootstrap calibration

```bash
python scripts/calibrate_online_conformal.py
python scripts/calibrate_online_conformal.py --quick   # 100 calls per stratum
```

Requires `models/cqr_yield.pt`. Uses CMIP6 + ERA5 Zarr when present; otherwise synthetic climates with ensemble-spread noise.

Re-run after refreshing `CMIP6_ZARR_PATH` or retraining CQR.

## Live API updates

Each `/simulate-scenario` call:

1. Resolves FDP region from `farm_location`.
2. Predicts CQR quantiles on SSP-adjusted climate (parallel to CASEJ MC bands).
3. Uses `current_yield` (tonnes/ha) as the **farm-reported proxy** for projected-yield conformity scores until dedicated yield observations are posted.
4. Updates `q_t` via ECI-Integral and appends to the rolling coverage deque.
5. Returns `confidence_interval.method` and `confidence_interval.coverage_running_avg`.

Financial impact bounds use the online conformal avoided-loss interval when conformal is active.

## Validation

```bash
python scripts/validate_scenario_coverage.py
python scripts/validate_scenario_coverage.py --synthetic   # score-space shift sim (CI)
```

Acceptance: ECI-Integral empirical 90% PI coverage ∈ [88%, 92%] for all 48 strata.

## Drift Detection: WCTM + Conformal CUSUM

Post-deployment monitoring uses **Weighted Conformal Test Martingales (WCTM)** per Prinster et al., ICML 2025 (WATCH), with a parallel **conformal CUSUM** sanity check (Vovk et al., PMLR 266).

### Behaviour

Each `/simulate-scenario` call (when `DRIFT_ENABLED=true` and online conformal is active):

1. Computes normalized nonconformity `|current_yield − y_pred| / σ_t` with `y_pred` = CQR median and `σ_t` from half the projected yield interval.
2. Updates label-WCTM and a lightweight X-CTM (climate/static feature EMA distance) per stratum.
3. Persists state to `data/processed/drift_monitoring_state.json` (or Redis key `drift_monitoring_state`).
4. Returns optional `drift_alarm` and `drift_status` on the response.
5. On **`concept_shift`** alarms, widens conformal intervals by inflating `q_t` with `DRIFT_INFLATION_FACTOR` (default 1.5).

### Stratum keys

Same as conformal: `{scenario}:{horizon_year}:{region}`.

### Dashboard API

```bash
curl "http://localhost:8000/drift-status?stratum=ssp245:2050:ghana"
```

### Validation

```bash
python scripts/validate_drift_detection.py
python scripts/validate_drift_detection.py --quick
python scripts/validate_drift_detection.py --plot   # dev: matplotlib figures only in script
```

Report: `reports/monitoring/wctm_validation_<date>.md`.

### Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `DRIFT_ENABLED` | `true` | Toggle WCTM updates |
| `DRIFT_STATE_PATH` | `data/processed/drift_monitoring_state.json` | Persisted martingale state |
| `DRIFT_ALPHA_FPR` | `0.01` | Ville threshold (~1 false alarm per 100 strata-updates under null) |
| `DRIFT_INFLATION_FACTOR` | `1.5` | ECI threshold multiplier on `concept_shift` |
| `DRIFT_SCORE_CAP` | `8.0` | `out_of_support` diagnosis |

## Environment

See `.env.example`: `CONFORMAL_METHOD`, `CONFORMAL_ALPHA`, `ECI_ETA`, `ECI_DECAY`, `REDIS_URL`, conformal and drift state paths.
