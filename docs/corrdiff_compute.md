# CorrDiff-CMIP6 scenario downscaling — deferred HPC workflow

> **Full avoided-loss stack:** see [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).

The **code paths** for CorrDiff-CMIP6 km-scale downscaling (`counterfactual.corrdiff_downscaler`, `POST /simulate-scenario` with `downscaling_method=corrdiff`) are in this repository. **Precomputed Zarr ensembles** under `data/processed/corrdiff_{ssp}_{horizon}_{region}.zarr` are **not** checked in until you run bulk inference on GPU/HPC.

`downscaling_method=linear_delta` (default) remains available on CPU-only machines.

## Hardware and install

```bash
pip install -e ".[corrdiff]"
```

- **GPU:** NVIDIA A100-80GB or H100 recommended.
- **Checkpoint:** `nvidia/corrdiff-cmip6-era5` on Hugging Face (~80 GB GPU memory at inference; Apache-2.0).
- **DVC:** track checkpoint under `models/corrdiff_cmip6/` (see `scripts/download_corrdiff_checkpoint.py`); never commit weights to git.

## When you have HPC access

### 1. Download checkpoint (once)

```bash
python scripts/download_corrdiff_checkpoint.py
```

### 2. Pre-compute 48 strata (2 SSP × 3 horizons × 8 FDP regions)

```bash
PYTHONPATH=src python scripts/run_corrdiff_scenario_bulk.py
```

**Expected runtime:** ~4 hours per stratum on H100 (~190 GPU-hours total). Use `--strata ssp245:2030:ghana` for a single stratum, `--dry-run` to list work, `--force` to rebuild.

Outputs:

| Artifact | Path |
|----------|------|
| Per-stratum ensemble | `data/processed/corrdiff_{scenario}_{horizon}_{region}.zarr` |
| Manifest | `data/processed/corrdiff_bulk_manifest.json` |

### 3. Validation (hindcast 2021–2024)

```bash
PYTHONPATH=src python scripts/validate_corrdiff_vs_linear_delta.py --quick   # smoke
PYTHONPATH=src python scripts/validate_corrdiff_vs_linear_delta.py           # full gate (HPC)
```

Report: `reports/scenarios/corrdiff_validation_<date>.md`

### 4. API

```json
POST /simulate-scenario
{
  "downscaling_method": "corrdiff",
  "scenario": "ssp245",
  "horizon_year": 2030,
  ...
}
```

Requires cached Zarr for the resolved stratum (`scenario:horizon:region`). Online conformal calibrates on a separate stratum key suffix `:corrdiff`.

## Local runs that did not finish (safe to stop)

| Command | Why it stalls | What was not produced |
|---------|---------------|------------------------|
| `run_corrdiff_scenario_bulk.py` (full 48) | ~190 GPU-hours | `corrdiff_*.zarr` caches |
| `validate_corrdiff_vs_linear_delta.py` (full) | Many CorrDiff forwards | Full validation report |

```bash
pkill -f "run_corrdiff_scenario_bulk.py" || true
pkill -f "validate_corrdiff_vs_linear_delta.py" || true
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CMIP6_ZARR_PATH` | Repo CMIP6 ensemble Zarr (GDDP-style) for linear baseline |
| `ERA5_ZARR_PATH` | Historical ERA5 for validation hindcast |
| `CORRDIFF_ALLOW_INLINE` | If `true`, allow on-demand CorrDiff when cache missing (lab only) |
| `CORRDIFF_PROCESSED_DIR` | Override `data/processed` cache root |
