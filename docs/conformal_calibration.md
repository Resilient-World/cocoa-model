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

## Environment

See `.env.example`: `CONFORMAL_METHOD`, `CONFORMAL_ALPHA`, `ECI_ETA`, `ECI_DECAY`, `REDIS_URL`, state paths.
