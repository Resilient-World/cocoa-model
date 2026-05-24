# Training runbook — resilient cocoa / avoided-loss model

This document is the **canonical checklist** for training everything the product needs: cocoa **exposure** (where cocoa is), **yield** and **uncertainty** (how much is lost or avoided), **scenario** simulation (CASEJ), and optional **causal** evaluation. None of these full training jobs are meant to run on a laptop CPU for production-quality artifacts.

## Local runs that did not finish (safe to stop)

If background jobs were started from Cursor or a terminal (for example `train_agrifm_cocoa.py` on CPU or `fit_ensemble_v2_weights.py`), they can be **stopped without losing repo state**:

| Typical command | Why it stalls on a laptop | What was *not* produced |
|-----------------|---------------------------|------------------------|
| `python scripts/train_agrifm_cocoa.py` (no `--synthetic`) | ~100M-param Video Swin, 50 epochs | `models/agrifm_cocoa_seg.pt` |
| `python scripts/fit_ensemble_v2_weights.py` | Thousands of tile forwards × 8 regions × grid search | Fitted `config/ensemble_weights.yaml` regions |
| `python -m training.train_galileo_cocoa` | Large multimodal encoder | `models/galileo_cocoa_seg.pt` |
| `python scripts/validate_dvds.py --reps 200 --n 1000` | ~200 × (DVDS + Zhao bootstrap B=500) per replication; multi-hour on laptop CPU | Production gate report `reports/sensitivity/dvds_validation_<date>.md` with full Section 7.1 coverage |
| `python scripts/run_corrdiff_scenario_bulk.py` (48 strata) | ~190 GPU-hours on H100 (~4 h/stratum) | `data/processed/corrdiff_{ssp}_{horizon}_{region}.zarr` + manifest |

**Stopping incomplete jobs does not require a git revert.** All training logic lives in the repository; only **checkpoints and fitted YAML** are missing until you rerun on GPU/HPC.

```bash
# Optional: find and stop stray training processes
pkill -f "train_agrifm_cocoa.py" || true
pkill -f "fit_ensemble_v2_weights.py" || true
pkill -f "validate_dvds.py" || true
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
| TerraMind cocoa | `pip install -e ".[terramind]"` then `python scripts/train_terramind_cocoa.py` | `models/terramind_cocoa_seg.pt` |
| Ensemble v2 weights | `python scripts/fit_ensemble_v2_weights.py` | `config/ensemble_weights.yaml` (per-region) |
| Ensemble v3 weights | `python scripts/fit_ensemble_v3_weights.py` | `config/ensemble_weights_v3.yaml` (NNLS, 5-way) |

Detailed exposure fine-tune + ensemble steps: [`docs/ensemble_v3_compute.md`](ensemble_v3_compute.md).

#### LoRA per-region adapters

New region onboarding should prefer LoRA adapters over full fine-tunes:

```bash
python -m training.train_galileo_cocoa --region ghana --lora
python scripts/train_agrifm_cocoa.py --region ghana --lora
python scripts/train_terramind_cocoa.py --region ghana --lora
python scripts/train_olmoearth_cocoa.py --region ghana --lora
python scripts/train_aef_head.py --region ghana --lora --synthetic
```

Adapters are saved as `models/<backbone>_lora_<region>.safetensors`. Use `scripts/benchmark_lora_vs_full_finetune.py` to compare F1, mIoU, checkpoint size, and training time; acceptance is at least 10× checkpoint reduction with no more than 2 pp F1 loss.

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

### CorrDiff-CMIP6 scenario downscaling (HPC only)

Optional km-scale CMIP6 downscaling for `/simulate-scenario` (`downscaling_method=corrdiff`). Default `linear_delta` works on CPU.

```bash
pip install -e ".[corrdiff]"
python scripts/download_corrdiff_checkpoint.py   # HF weights → models/corrdiff_cmip6/
PYTHONPATH=src python scripts/run_corrdiff_scenario_bulk.py --strata ssp245:2030:ghana
PYTHONPATH=src python scripts/validate_corrdiff_vs_linear_delta.py --quick
```

Hardware: **A100-80GB or H100**; checkpoint `nvidia/corrdiff-cmip6-era5`. See [`corrdiff_compute.md`](corrdiff_compute.md).

```bash
pkill -f "run_corrdiff_scenario_bulk.py" || true
pkill -f "validate_corrdiff_vs_linear_delta.py" || true
```

**DVDS MSM validation (deferred on laptop):** Implementation and unit tests are merged; the **full coverage gate** is not run locally until you have more compute (workstation or HPC). Smoke only:

```bash
PYTHONPATH=src python scripts/validate_dvds.py --reps 5 --n 500
```

Production gate (Section 7.1 binary DGP, 200 replications, Zhao bootstrap comparison):

```bash
PYTHONPATH=src python scripts/validate_dvds.py --reps 200 --n 1000
```

See [`docs/sensitivity.md`](sensitivity.md) and commit the generated `reports/sensitivity/dvds_validation_<date>.md` after the full run.

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
dvc repro stage_train_terramind_cocoa
dvc repro stage_train_aef_head
dvc repro stage_train_joint
```

Full pipeline requires prior ingest and GPU stages; see `dvc.yaml`.

---

## Demo v0.3.0 CPU cost (offline)

| Module | Typical laptop time | Env / notes |
|--------|---------------------|-------------|
| **Mediation** (`decompose_mediators` × 3) | ~30–90 s | `MEDIATION_N_BOOTSTRAP=200` (API default); full 500 in offline scripts |
| **Policy tree** (demo panel, `n_bootstrap=0`) | ~5–15 s | Synthetic 400-farm panel only |
| **DVDS** (`include_sensitivity`) | ~2–5 s | Uses synthetic panel fallback when no parquet |
| **WCTM drift** on scenario | &lt;1 s | Empty drift store until `validate_drift_detection.py` seeds strata |
| **CorrDiff scenario** | skipped by default | Set `CORRDIFF_AVAILABLE=true` and precompute Zarr via `run_corrdiff_scenario_bulk.py` |

```bash
USE_REAL_FEATURES=false python scripts/demo_end_to_end.py --mock-gee --pretty
# outputs: reports/demo/e2e_civ_v5.json + e2e_civ_v5.md
```

---

## Related docs

- [`mediation_analysis.md`](mediation_analysis.md) — NDE/NIE, mediator IDs, ρ sensitivity
- [`ensemble_v3_compute.md`](ensemble_v3_compute.md) — AgriFM + TerraMind fine-tune + ensemble v2/v3
- [`MODEL_CARD.md`](MODEL_CARD.md) — thresholds, regulatory mapping
- [`README.md`](../README.md) — architecture and API
