# Training runbook — resilient cocoa / avoided-loss model

This document is the **canonical checklist** for training everything the product needs: cocoa **exposure** (where cocoa is), **yield** and **uncertainty** (how much is lost or avoided), **scenario** simulation (CASEJ), and optional **causal** evaluation. None of these full training jobs are meant to run on a laptop CPU for production-quality artifacts.

## Local runs that did not finish (safe to stop)

If background jobs were started from Cursor or a terminal (for example `train_agrifm_cocoa.py` on CPU or `fit_ensemble_v2_weights.py`), they can be **stopped without losing repo state**:

| Typical command | Why it stalls on a laptop | What was *not* produced |
|-----------------|---------------------------|------------------------|
| `python scripts/train_agrifm_cocoa.py` (no `--synthetic`) | ~100M-param Video Swin, 50 epochs | `models/agrifm_cocoa_seg.pt` |
| `python scripts/fit_ensemble_v2_weights.py` | Thousands of tile forwards × 8 regions × grid search | Fitted `config/ensemble_weights.yaml` regions |
| `python -m training.train_galileo_cocoa` | Large multimodal encoder | `models/galileo_cocoa_seg.pt` |

**Stopping incomplete jobs does not require a git revert.** All training logic lives in the repository; only **checkpoints and fitted YAML** are missing until you rerun on GPU/HPC.

```bash
# Optional: find and stop stray training processes
pkill -f "train_agrifm_cocoa.py" || true
pkill -f "fit_ensemble_v2_weights.py" || true
```

CI and DVC smoke stages use `--synthetic` / `--quick` and are the only training expected to pass on CPU.

---

## Compute guidance

| Tier | Hardware | Suitable for |
|------|----------|--------------|
| **Smoke** | Laptop CPU, 16 GB RAM | `pytest`, `--synthetic` scripts, `dvc repro` synthetic stages |
| **Exposure training** | 1× GPU ≥ 16 GB VRAM | AgriFM / Galileo fine-tune, AEF head, ensemble weight fit |
| **Full product** | GPU + GEE / large Zarr disk | ERA5 ingest, yield + CQR, joint model, CMIP6 scenarios |

Authenticate Earth Engine before any GEE-backed ingest or Kalischek validation:

```bash
source .venv/bin/activate
earthengine authenticate   # or service account via GOOGLE_APPLICATION_CREDENTIALS
export EARTHENGINE_PROJECT=your-gcp-project-id
python -m data.gee_auth
```

---

## End-to-end order (production artifacts)

Run from the repository root. Stages marked **(GPU)** should not be skipped for production.

### Phase 0 — Data ingest (GEE / disk)

| Step | Command | Output |
|------|---------|--------|
| AOI + manifest | `python scripts/ingest_gee.py` | `data/raw/ingest_manifest.json`, `cocoa_aoi.geojson` |
| ERA5 stack | `python -m data.era5_ingest --aoi data/raw/cocoa_aoi.geojson --out data/processed/era5_2020_2024.zarr` | Climate Zarr for API + yield |
| Sentinel composites | `python -m data.sentinel_composite --destination local ...` | Tiles under `data/raw/` / `processed/` |
| CMIP6 (scenarios) | `python -m data.cmip6_ingest ...` | `data/processed/cmip6_ensemble.zarr` |

Set paths in `.env` (`ERA5_ZARR_PATH`, `CMIP6_ZARR_PATH`, `STATIC_ZARR_PATH`).

### Phase 1 — Cocoa exposure / segmentation **(GPU)**

These models feed `cocoa_prob` in the API and `ensemble_v2` exposure.

| Model | Train | Checkpoint / config |
|-------|-------|---------------------|
| FDP 2025a | GEE asset (no train) | `projects/forestdatapartnership/assets/cocoa/model_2025a` |
| AlphaEarth head | `python scripts/train_aef_head.py` | `models/aef_cocoa_head.pt` |
| Galileo seg | `python -m training.train_galileo_cocoa` | `models/galileo_cocoa_seg.pt` |
| AgriFM backbone | `python scripts/download_agrifm_weights.py` | `models/agrifm/agrifm_s2_pretrained.pt` |
| AgriFM cocoa | `python scripts/train_agrifm_cocoa.py --pretrained models/agrifm/agrifm_s2_pretrained.pt` | `models/agrifm_cocoa_seg.pt` |
| Ensemble v2 weights | `python scripts/fit_ensemble_v2_weights.py` | `config/ensemble_weights.yaml` (per-region) |

Detailed AgriFM + ensemble steps: [`docs/agrifm_ensemble_v2_compute.md`](agrifm_ensemble_v2_compute.md).

Benchmark after checkpoints exist:

```bash
python scripts/benchmark_backbones.py --write-ensemble-v2-report
```

**Production gate:** `fit_ensemble_v2_weights.py` should report ensemble F1 ≥ best single backbone on **≥ 6 / 8** FDP regions before treating `COCOA_EXPOSURE_BACKEND=ensemble_v2` as live-default.

### Phase 2 — Avoided-loss yield stack **(GPU)**

Powers `POST /simulate-intervention` (baseline vs projected yield → avoided tonnes → USD).

| Model | Train | Checkpoint |
|-------|-------|------------|
| Yield surrogate | `python -m training.train_yield --config-name yield` or `python scripts/train_yield_surrogate.py` | `models/yield_surrogate_v1.pt` |
| CQR calibration | `python scripts/train_cqr.py` | `models/cqr_yield.pt`, `models/cqr_calibrator.joblib` |
| Joint exposure+yield (optional) | `python scripts/train_joint.py` | `models/joint.pt` |

API env: `MODEL_CHECKPOINT_PATH`, `UQ_METHOD=cqr`, `CQR_*_PATH` (see `.env.example`).

### Phase 3 — Climate scenario / CASEJ **(GPU)**

Powers `POST /simulate-scenario` (CMIP6-adjusted ERA5 + CO₂-aware physiology).

```bash
python scripts/generate_casej_training_set.py
python scripts/train_casej_surrogate.py
# → models/casej_surrogate.pt (CASEJ_CHECKPOINT_PATH)
```

Requires `ERA5_ZARR_PATH` and `CMIP6_ZARR_PATH`.

### Phase 4 — Causal evaluation (CPU OK; needs panel data)

Research / cohort **observational** avoided revenue (not the same code path as the API intervention simulator):

```bash
python -m analysis.run_evaluation --panel data/raw/farm_panel.parquet --out reports/causal_eval.json
```

Uses PSM + DiD (`src/analysis/`).

### Phase 5 — Verify product path

```bash
# Mock features (no GEE)
USE_REAL_FEATURES=false python scripts/demo_end_to_end.py --mock-gee --out reports/demo/e2e.json

# With real Zarr + checkpoints + GEE
python scripts/demo_end_to_end.py --out reports/demo/e2e_civ.json
uvicorn api.main:app --reload
# POST /simulate-intervention
```

---

## What to commit after GPU training (follow-up PRs)

Large binaries are usually **DVC-tracked** or attached to releases, not every commit:

| Artifact | DVC / git |
|----------|-----------|
| `models/agrifm_cocoa_seg.pt` | DVC `outs` or release asset |
| `models/galileo_cocoa_seg.pt` | DVC |
| `models/aef_cocoa_head.pt` | DVC (`stage_train_aef_head`) |
| `config/ensemble_weights.yaml` (fitted regions) | **Git** (small YAML) |
| `models/yield_surrogate_v1.pt`, CQR, CASEJ, joint | DVC stages in `dvc.yaml` |
| `data/processed/*.zarr` | DVC only |

Until fitted weights and AgriFM checkpoint exist, keep API default exposure on `fdp` or `ensemble` (v1) in `.env` if you need stable demos.

---

## DVC quick reference

Synthetic smoke (laptop-friendly):

```bash
dvc repro stage_train_agrifm_cocoa
dvc repro stage_train_aef_head
dvc repro stage_train_joint
```

Full pipeline requires prior ingest and GPU stages; see `dvc.yaml`.

---

## Related docs

- [`agrifm_ensemble_v2_compute.md`](agrifm_ensemble_v2_compute.md) — AgriFM fine-tune + ensemble v2 only
- [`MODEL_CARD.md`](MODEL_CARD.md) — thresholds, regulatory mapping
- [`README.md`](../README.md) — architecture and API
