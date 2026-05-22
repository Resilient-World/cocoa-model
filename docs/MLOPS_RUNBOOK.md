# MLOps runbook — resilient cocoa-model v0.3.0

Canonical flow for reproducible training, hyperparameter search, champion/challenger promotion, and container release.

See also: [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) (GPU/GEE production training), [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

```bash
cd resilient-cocoa-model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mlops]"
export PYTHONPATH=src
```

Optional: local MLflow tracking

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow server --host 127.0.0.1 --port 5000
```

## 1. DVC pipeline (`params.yaml` + `dvc.yaml`)

`params.yaml` sets `mock_gee: true` for CPU/CI smoke (stub ERA5/S2/FDP, synthetic training).

```bash
dvc repro                    # full graph
make dvc-repro               # same
dvc repro -s stage_eval_e2e  # single stage
```

**Production** (`mock_gee: false`): requires GEE auth and GPU; see TRAINING_RUNBOOK.

| Stage | Output |
|-------|--------|
| `stage_ingest_era5` | `data/processed/era5_2020_2024.zarr` |
| `stage_ingest_s2` | `data/processed/s2_s1_manifest.json` |
| `stage_ingest_fdp` | `data/raw/fdp_ingest_manifest.json` |
| `stage_train_*` | Segmentation + yield + CQR checkpoints |
| `stage_fit_ensemble_v3` | `config/ensemble_weights_v3.yaml` |
| `stage_validate` | `reports/validation/summary.md` |
| `stage_eval_e2e` | `reports/demo/e2e_civ_v5.json` |

## 2. Optuna HPO → MLflow

Experiments: `hpo_yield`, `hpo_galileo`, `hpo_agrifm`.

```bash
make hpo MODEL=yield N=50
make hpo MODEL=galileo N=30
python scripts/run_hpo.py --model agrifm --n-trials 100
```

Yield HPO optimizes **CRPS** on a spatial holdout fold (`data.spatial_splits`).

## 3. Champion / challenger registry

Register a training run as **challenger**:

```bash
python scripts/register_checkpoint.py \
  --model-name yield_surrogate_v2 \
  --checkpoint models/yield_surrogate_v2.pt
```

Promote after gates (CRPS, CQR coverage 88–92% @ 90%, `/simulate-intervention` smoke, EUDR+Whisp):

```bash
make promote PROMOTE_MODEL=yield_surrogate_v2 PROMOTE_RUN_ID=<mlflow_run_id>
# or
scripts/promote_champion.sh yield_surrogate_v2 <run_id>
```

API loads `models:/yield_surrogate_v2@champion` when:

```bash
export MLFLOW_REGISTRY_ENABLED=true
export MLFLOW_REGISTRY_MODEL_NAME=yield_surrogate_v2
```

Falls back to `MODEL_CHECKPOINT_PATH` if the registry is empty.

Rollback:

```python
from registry.mlflow_registry import rollback
rollback("yield_surrogate_v2")
```

## 4. Release evidence

On promotion, artifacts are written to:

- `release_evidence/promotion_decision.json`
- `release_evidence/release_manifest.json`
- `release_evidence/rollback_target.json`
- `release_evidence/model_card.md`

Mirrored under `reports/releases/<model>/<env>/v<N>/`.

## 5. Container & CD

**Build** (push to `main` or tag `v*`):

```bash
docker build -t ghcr.io/resilient-world/cocoa-model:local .
```

GitHub Actions: `.github/workflows/build-image.yml` — pushes `ghcr.io/resilient-world/cocoa-model:{sha,latest,vX.Y.Z}` and **cosign** signs images.

**Staging CD** on tag `v*`: `.github/workflows/cd-staging.yml` runs promotion gate and updates the `champion` alias (requires `MLFLOW_TRACKING_URI` secret).

## 6. Quick verification

```bash
make lint typecheck test
pytest tests/registry/ -q
dvc repro -s stage_train_yield_v2
```

## License boundary

MLOps code is MIT. ATTRICI remains GPLv3 subprocess-only — promotion gates do not import ATTRICI modules.
