# resilient-cocoa-model

Geospatial machine learning and causal impact modeling for **resilient cocoa** in West Africa. This repository supports the [Resilient World](https://github.com/Resilient-World) platform by combining:

1. **Remote sensing** — Sentinel-1/2 composites, ERA5 climate, and Prithvi-based plantation mapping (TorchGeo / TerraTorch).
2. **Yield modeling** — A fast neural surrogate for process-based yield with Monte Carlo uncertainty.
3. **Causal impact** — Propensity score matching (PSM) and difference-in-differences (DiD) on farm panels.
4. **Product API** — FastAPI service so the Resilient frontend can **simulate interventions** and estimate avoided yield loss and financial impact.

**GitHub:** [Resilient-World/cocoa-model](https://github.com/Resilient-World/cocoa-model)

---

## Table of contents

- [Who this README is for](#who-this-readme-is-for)
- [System overview](#system-overview)
- [Repository layout](#repository-layout)
- [Technology stack](#technology-stack)
- [Getting started](#getting-started)
  - [Quickstart Demo](#quickstart-demo)
- [Configuration](#configuration)
- [Package reference](#package-reference)
  - [`src/data` — ingestion and datasets](#srcdata--ingestion-and-datasets)
  - [`src/models` — yield surrogate](#srcmodels--yield-surrogate)
  - [`src/training` — Prithvi fine-tuning](#srctraining--prithvi-fine-tuning)
  - [`src/analysis` — causal inference](#srcanalysis--causal-inference)
  - [`src/api` — intervention simulation service](#srcapi--intervention-simulation-service)
- [End-to-end workflows](#end-to-end-workflows)
- [Training runbook (GPU / deferred)](#training-runbook-gpu--deferred)
- [Testing](#testing)
- [Data versioning (DVC)](#data-versioning-dvc)
- [Design decisions and limitations](#design-decisions-and-limitations)
- [Contributing](#contributing)

---

## Who this README is for

New engineers joining the cocoa modeling effort should use this document to understand:

- **What** each directory and module does.
- **How** the geospatial pipeline, yield model, causal analysis, and API relate (and where they are still decoupled).
- **Where** to run code, tests, and the HTTP service.

The project is developed as a **standalone Python package** (typically at `~/resilient-cocoa-model`), separate from other Resilient monorepos.

---

## System overview

The codebase implements **two complementary tracks** that will increasingly converge as real checkpoints and live geospatial feeds replace mocks.

```mermaid
flowchart TB
    subgraph ingest [Data ingestion - Google Earth Engine]
        GEE[gee_auth]
        ERA5[era5_ingest]
        S2[sentinel_composite]
    end

    subgraph geoML [Geospatial ML - mapping]
        DS[cocoa_dataset]
        TR[train_prithvi_cocoa]
        CKPT_SEG[Segmentation checkpoints]
    end

    subgraph yieldML [Yield ML - surrogate]
        YSM[YieldSurrogateModel]
        MC[predict_with_uncertainty]
        CKPT_YLD[Yield checkpoints]
    end

    subgraph causal [Causal impact - farm panels]
        PSM[propensity_score_match]
        DID[calculate_did_att]
        REV[calculate_avoided_revenue_loss]
    end

    subgraph product [Product API]
        API[FastAPI /simulate-intervention]
    end

    GEE --> ERA5
    GEE --> S2
    S2 --> DS
    DS --> TR
    TR --> CKPT_SEG

    ERA5 -.->|future wiring| YSM
    YSM --> MC
    CKPT_YLD --> YSM

    PSM --> DID
    DID --> REV

    YSM --> API
    API -->|Resilient frontend| FE[Frontend platform]
```

| Track | Purpose | Primary outputs |
|-------|---------|-----------------|
| **Geospatial ML** | Map cocoa plantations (full-sun vs agroforestry) from Sentinel composites | GeoTIFF tiles, Prithvi segmentation weights |
| **Yield surrogate** | Fast yield prediction from daily climate + static site features | Tonnes/ha + epistemic uncertainty (MC dropout) |
| **Causal analysis** | Rigorous impact evaluation on observational farm panels | ATT (tonnes/ha), cohort avoided revenue (USD) |
| **API** | Prospective single-farm simulation for product UX | Baseline/projected yield, avoided loss, 90% CI, USD impact |

---

## Repository layout

```
resilient-cocoa-model/
├── data/                          # Large artifacts (gitignored; track with DVC)
│   ├── raw/                       # Immutable source inputs (GEE exports, etc.)
│   ├── interim/                   # Intermediate transforms
│   ├── processed/                 # Model-ready rasters / tables
│   └── external/                  # Third-party reference data
├── models/                        # Serialized checkpoints (DVC-tracked; not in git)
├── notebooks/                     # Exploratory analysis (Jupyter)
├── scripts/
│   ├── setup_venv.sh              # Recreate .venv, pip install -e ".[dev]", run tests
│   └── init_project.sh            # Scaffold dirs + git/DVC hints (no pip install)
├── src/                           # Installable package root (see pyproject.toml)
│   ├── data/                      # GEE auth, ERA5, Sentinel, TorchGeo datasets
│   ├── models/                    # YieldSurrogateModel + MC dropout
│   ├── training/                  # Prithvi fine-tuning (Lightning + TerraTorch)
│   ├── analysis/                  # PSM + DiD + financial valuation
│   └── api/                       # FastAPI Avoided Loss simulation service
├── tests/                         # pytest suite (mirrors src/ modules)
├── .dvc/                          # DVC metadata (pipeline stages empty for now)
├── dvc.yaml                       # DVC pipeline definition (extend as steps stabilize)
├── pyproject.toml                 # Package metadata, dependencies, pytest/ruff config
├── requirements.txt               # Pinned-style dependency list (mirrors pyproject)
├── .env.example                   # Environment variable template (never commit .env)
├── resilient-cocoa-model.code-workspace   # VS Code / Cursor workspace settings
└── README.md                      # This file
```

**Import convention:** After `pip install -e .`, import from top-level packages (`data`, `models`, `api`, …) because `package-dir = { "" = "src" }` in `pyproject.toml`.

---

## Technology stack

| Layer | Libraries |
|-------|-----------|
| Geospatial I/O | `geopandas`, `rasterio`, `xarray`, `netcdf4` |
| Earth Engine | `earthengine-api` (optional `geemap` via `[gee]` extra) |
| Deep learning | `torch`, `torchgeo`, `terratorch`, `lightning` (via TerraTorch) |
| Tabular / ML | `pandas`, `scikit-learn` |
| MLOps | `mlflow`, `dvc` |
| API | `fastapi`, `uvicorn`, `pydantic-settings`, `httpx` (tests) |
| Dev | `pytest`, `ruff`, `jupyter` |

**Python:** 3.10+ (development and CI use **3.12** via `setup_venv.sh`).

---

## Getting started

### Open the project

- **Cursor / VS Code:** File → Open Folder → `~/resilient-cocoa-model`, or open `resilient-cocoa-model.code-workspace`.
- Clone: `git clone https://github.com/Resilient-World/cocoa-model.git`

### One-command environment setup

```bash
cd ~/resilient-cocoa-model
./scripts/setup_venv.sh
source .venv/bin/activate
```

`setup_venv.sh` will:

1. Find Python 3.10+ (prefers 3.12).
2. Recreate `.venv` and upgrade `pip`.
3. Run `pip install -e ".[dev]"`.
4. Verify critical imports (`torch`, `torchgeo`, `fastapi`, …).
5. Run `pytest`.

### Manual setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

Optional Earth Engine helpers:

```bash
pip install -e ".[gee]"
```

### Earth Engine authentication

Required for `src/data` GEE scripts:

```bash
earthengine authenticate
export EARTHENGINE_PROJECT=your-gcp-project-id   # or set in .env
python -m data.gee_auth
```

`gee_auth.py` runs a Ghana test point elevation query (SRTM) to verify connectivity.

### Quickstart Demo

Run the full product pipeline on a sample Côte d'Ivoire farm polygon and emit one JSON summary (no GEE credentials required):

```bash
python scripts/demo_end_to_end.py --mock-gee --pretty
```

Output: `reports/demo/e2e_civ.json` with `climate_attributed_loss_t_per_ha`, `intervention_avoided_loss_t_per_ha`, `total_avoided_loss_usd` (90% interval), `eudr_status`, and `source_attributions` for every dataset. With Earth Engine configured, omit `--mock-gee` for live FDP / ERA5 sampling.

```bash
pytest tests/test_demo_end_to_end.py -q
```

See [CHANGELOG.md](CHANGELOG.md) for release notes (v0.2.0).

---

## Configuration

Copy `.env.example` to `.env` (never commit `.env`).

| Variable | Used by | Description |
|----------|---------|-------------|
| `EARTHENGINE_PROJECT` | GEE scripts | GCP project with Earth Engine enabled |
| `GOOGLE_APPLICATION_CREDENTIALS` | GEE (headless) | Service account JSON path |
| `MLFLOW_TRACKING_URI` | Prithvi training | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | Prithvi training | Experiment name |
| `API_HOST` / `API_PORT` | Deployment | Uvicorn bind (documentation; not auto-wired in code) |
| `MODEL_CHECKPOINT_PATH` | API | Path to `YieldSurrogateModel` weights (`.pt`) |
| `CASEJ_CHECKPOINT_PATH` | API | Path to `CASEJSurrogate` weights for `/simulate-scenario` |
| `MC_NUM_SAMPLES` | API | Monte Carlo forward passes (default `50`) |
| `YIELD_BLEND_WEIGHT` | API | Blend weight for observed `current_yield` (default `0.3`) |
| `ERA5_ZARR_PATH` | API | Historical ERA5-Land stack (`ScenarioBuilder` + `/simulate-intervention` resolver) |
| `CMIP6_ZARR_PATH` | API | NASA/GDDP-CMIP6 (or compatible) ensemble Zarr for `/simulate-scenario` deltas |
| `DVC_REMOTE_URL` | DVC | Example remote store URL |

API settings are loaded via `pydantic-settings` in `api.config.APISettings`.

---

## Package reference

### `src/data` — ingestion and datasets

| Module | CLI | Role |
|--------|-----|------|
| [`gee_auth.py`](src/data/gee_auth.py) | `python -m data.gee_auth` | Initialize EE; verify Ghana elevation |
| [`era5_ingest.py`](src/data/era5_ingest.py) | `python -m data.era5_ingest --region ghana …` | ERA5 daily Zarr per region (`data/processed/era5_<region>_<years>.zarr`) |
| [`sentinel_composite.py`](src/data/sentinel_composite.py) | `python -m data.sentinel_composite --region civ` | Cloud-masked S2/S1 composite → `data/processed/s2_s1_<region>.tif` |
| [`cocoa_dataset.py`](src/data/cocoa_dataset.py) | — | TorchGeo `CocoaImagery` / `CocoaMask` / `CocoaDataset` + `CocoaDataModule` |
| [`cocoa_exposure.py`](src/data/cocoa_exposure.py) | — | FDP cocoa probability (2025a) via GEE; point sample + optional Zarr export |

#### Cocoa exposure layer

[`cocoa_exposure.py`](src/data/cocoa_exposure.py) ingests the Forest Data Partnership **Cocoa Probability model 2025a** — successor to Kalischek et al. (2023, *Nature Food*) — as a server-side Earth Engine ImageCollection (no raw tile downloads).

| Item | Value |
|------|--------|
| Asset | `projects/forestdatapartnership/assets/cocoa/model_2025a` (`FDP_COCOA_COLLECTION`) |
| Resolution | 10 m |
| Years | 2020, 2023 (annual composites) |
| Band | `probability` (0–1 cocoa occupancy) |
| Default threshold | **0.96** ([FDP 2025a model card](https://github.com/google/forest-data-partnership/tree/main/models/cocoa) F1-optimal precision/recall) |
| Coverage | Ghana, Côte d'Ivoire, Cameroon, Nigeria, Indonesia, Ecuador, Peru, Colombia (`REGIONS` in `cocoa_exposure.py`) |

**Regions:** `data.cocoa_exposure.REGIONS` defines bounding-box `ee.Geometry` presets for each country. CLI: `--region ghana|civ|cameroon|nigeria|indonesia|ecuador|peru|colombia` on `era5_ingest` and `sentinel_composite`.

**API:** `CocoaExposureIngest(aoi, year=2023, threshold=0.96)` exposes `probability_image()`, `binary_mask()`, `sample_point()`, `to_zarr()` (Xee), and `area_hectares()`.

**Product wiring:** `api/feature_resolver.py` calls `sample_cocoa_probability_at_point()` (static index 9). Inside FDP-native `REGIONS`, FDP (or configured backend) is used; **outside** those bounds, exposure falls back to **AlphaEarth embeddings + Galileo** (globally available). Masked FDP pixels still use the belt heuristic.

**License:** Non-commercial Earth Engine use — **CC-BY 4.0 NC** with attribution *"Produced by Google for the Forest Data Partnership"*. **Commercial / SaaS deployments** must accept the [Forest Data Partnership Datasets Commercial Terms of Use](https://services.google.com/fh/files/misc/forest_data_partnership_datasets_commerical_terms_of_use.pdf).

Configure defaults via `.env` (`COCOA_EXPOSURE_YEAR`, `COCOA_EXPOSURE_THRESHOLD`).

#### Backbone choice (cocoa exposure refinement)

Segmentation backbones are compared on a **5000-tile held-out** set per FDP region (or all regions) with Kalischek et al. (2023) in-situ labels. Run:

```bash
python scripts/benchmark_backbones.py --n-tiles 5000 --region ghana
python scripts/benchmark_backbones.py --all-regions --quick   # 200 tiles/region, Galileo-nano
python scripts/benchmark_backbones.py --quick                 # 200 tiles, all regions pooled
python scripts/benchmark_backbones.py --backbone agrifm --region ghana --quick
python scripts/download_agrifm_weights.py                   # S2 pretrained weights (Apache-2.0)
```

Latest reports: [`reports/backbones/benchmark_<region>_<date>.md`](reports/backbones/) and [`benchmark_aef_<region>_<date>.md`](reports/backbones/) (mean error + mIoU/F1). AgriFM-only runs write [`benchmark_agrifm_<region>_<date>.md`](reports/backbones/).

| Role | Backbone | Notes |
|------|----------|--------|
| **Production (candidate)** | **AlphaEarth Foundations (AEF)** + MLP head | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` 64-D annual embeddings (arXiv:2507.22291); near-zero inference via [`alphaearth_embeddings.py`](src/data/alphaearth_embeddings.py) + [`aef_cocoa_head.py`](src/models/aef_cocoa_head.py) |
| **Production (active)** | **Galileo + AlphaEarth + AgriFM ensemble (region-weighted)** | `ensemble_v2` loads per-region weights from [`config/ensemble_weights.yaml`](config/ensemble_weights.yaml) (fit via [`scripts/fit_ensemble_v2_weights.py`](scripts/fit_ensemble_v2_weights.py)); blends AEF, Galileo-Base, AgriFM Video Swin, and FDP |
| **Segmentation (single)** | **Galileo-Base** + binary seg head | Multimodal S2×10, S1 VV/VH, ERA5 monthly (5 vars → Galileo time bands), DEM; 10 m `P(cocoa)` via [`galileo_seg.py`](src/models/galileo_seg.py) |
| **Foundation (agriculture)** | **AgriFM (Video Swin)** | Li et al. (RSE 2026; arXiv:2505.21357); S2 10-band temporal stack; MIT reimplementation in [`agrifm_backbone.py`](src/models/agrifm_backbone.py) + [`agrifm_cocoa_head.py`](src/models/agrifm_cocoa_head.py); **Apache-2.0** pretrained weights via [`download_agrifm_weights.py`](scripts/download_agrifm_weights.py) |
| **Prior (weak supervision)** | FDP 2025a | GEE `model_2025a`; threshold 0.96 |
| **Baseline** | Prithvi-EO-2.0 | TerraTorch `prithvi_eo_v2_100_tl` (6-band stem in benchmark when full checkpoint absent) |

**Exposure API:** `CocoaExposureIngest(..., backend="fdp" | "galileo" | "aef" | "agrifm" | "ensemble" | "ensemble_v2")`. Production default: **`ensemble_v2`** (`COCOA_EXPOSURE_BACKEND=ensemble_v2`, `ENSEMBLE_BACKEND=v2`). Legacy v1 ensemble: `0.5 × AEF + 0.3 × Galileo + 0.2 × FDP`. Checkpoints: `models/aef_cocoa_head.pt`, `models/galileo_cocoa_seg.pt`, `models/agrifm_cocoa_seg.pt` (`python scripts/train_agrifm_cocoa.py`), weights YAML via `python scripts/fit_ensemble_v2_weights.py`.

> **Note:** Production training (exposure fine-tunes, yield/CQR, ensemble weights, ERA5 ingest) requires **GPU or HPC** and is not run on a laptop in CI. See **[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md)** for the full avoided-loss model checklist; AgriFM + ensemble v2 detail is in [`docs/agrifm_ensemble_v2_compute.md`](docs/agrifm_ensemble_v2_compute.md). Incomplete local training jobs can be stopped without losing code—only checkpoints are missing until you rerun on stronger compute.

**ERA5 outputs (per pixel, per year):**

- Heat-stress day count: days with daily max 2 m temperature > 32 °C.
- Annual total precipitation.

**Sentinel composite bands (default):** `B2`–`B12`, `NDVI`, `EVI`, `S1_VV`, `S1_VH` (10 m export).

**Land-cover classes (masks):**

| ID | Name |
|----|------|
| 0 | other |
| 1 | full_sun_cocoa |
| 2 | agroforestry_cocoa |

Exports support `drive` or `local` destinations (shared helpers in `era5_ingest`).

#### CSSVD landscape incidence (Dumont et al. 2025)

[`cssvd_landscape.py`](src/hazards/cssvd_landscape.py) predicts **12-month CSSVD incidence** from landscape and climate covariates (ensemble cocoa probability, 500 m non-cocoa buffer, 1 km canopy fragmentation, CHIRPS extreme precipitation, ERA5 growing-season diurnal range, Muller 2018 strain region). Training uses scikit-survival CoxBoost-style boosting on the Dumont supplement joined to [`cssvd_landscape_features.py`](src/data/cssvd_landscape_features.py).

```bash
# Place journal supplement at data/external/dumont_cssvd_plots.csv (or use synthetic CI data)
python scripts/train_cssvd_landscape.py --synthetic --checkpoint models/cssvd_landscape.joblib
# Optional: precompute GEE features for all plots
python scripts/train_cssvd_landscape.py --use-gee

export ENABLE_CSSVD_LANDSCAPE=true
export CSSVD_LANDSCAPE_CHECKPOINT=models/cssvd_landscape.joblib
```

When enabled, [`composite.py`](src/hazards/composite.py) maps predicted incidence to Ofori et al. yield loss; otherwise static `cssvd_prevalence_pct` is used. DVC stage: `stage_train_cssvd_landscape` in [`dvc.yaml`](dvc.yaml).

---

### `src/models` — yield surrogate

[`yield_surrogate.py`](src/models/yield_surrogate.py) implements a **two-branch** predictor:

- **LSTM** over daily climate `[batch, 365, 4]` (max/min temp, precip, radiation).
- **MLP** over static site features `[batch, 10]`.
- **Fusion head** → scalar yield (tonnes/ha).

| Symbol | Description |
|--------|-------------|
| `YieldSurrogateModel` | Main `nn.Module` |
| `MCDropout` | Dropout active at inference for epistemic uncertainty |
| `predict_with_uncertainty()` | N stochastic forwards → `mean`, `std` |
| `PhysicsInformedYieldLoss` | MSE + penalty when prediction exceeds biophysical `y_max` |

Designed as a fast surrogate for slow process models (e.g. ALMANAC-style simulators). **Not yet wired** to live ERA5 tensors from `era5_ingest` in production code paths.

**Conformal yield UQ:** [`cqr.py`](src/models/cqr.py) provides split-conformal CQR (`ConformalCalibrator`). For **distribution shift** (e.g. CMIP6 deltas on `/simulate-scenario`), use online methods in [`aci.py`](src/models/aci.py), [`conformal_pid.py`](src/models/conformal_pid.py), [`eci.py`](src/models/eci.py), and [`quantile_yield_surrogate_online.py`](src/models/quantile_yield_surrogate_online.py). `MultiStepACI` stratifies thresholds by horizon (`2030`, `2050`, `2080`). Benchmark: `python -m models.validate_conformal_coverage --benchmark-online`.

---

### `src/training` — Prithvi fine-tuning

| Module | CLI | Role |
|--------|-----|------|
| [`train_prithvi_cocoa.py`](src/training/train_prithvi_cocoa.py) | `python -m training.train_prithvi_cocoa` | Fine-tune NASA-IBM Prithvi via TerraTorch `SemanticSegmentationTask` |
| [`cocoa_prithvi_datamodule.py`](src/training/cocoa_prithvi_datamodule.py) | — | `CocoaPrithviDataModule` — Prithvi band mapping + normalization |

**Defaults:**

- Backbone: `prithvi_eo_v2_100_tl`
- Decoder: `UperNetDecoder` (FPN-style; `UNetDecoder` optional)
- Logging: MLflow via PyTorch Lightning
- Optimizer: AdamW + cosine LR schedule

Expects paired imagery/mask GeoTIFF directories compatible with `CocoaDataModule`.

---

### `src/analysis` — causal inference

Observational impact evaluation on **farm-level panels** (pandas). Used for rigorous cohort analysis, distinct from the prospective API simulator.

#### Propensity score matching & DML — [`psm_matching.py`](src/analysis/psm_matching.py)

| Function | Description |
|----------|-------------|
| `compute_propensity_scores()` | `StandardScaler` + logistic regression → P(treatment \| covariates) |
| `default_logit_caliper()` | Rosenbaum–Rubin caliper: `0.2 × SD(logit PS)` |
| `match_nearest_neighbor()` | k:1 NN on raw or logit propensity; optional replacement |
| `trim_common_support()` | Restrict to overlap region |
| `standardized_mean_differences()` | SMD before/after matching → `BalanceReport` (\|SMD\| &lt; 0.1 gate) |
| `love_plot_data()` | Long-format SMD table for Love plots / MLflow artifacts |
| `aipw_estimator()` | K-fold cross-fit DML/AIPW (HistGradientBoosting nuisances); ATE + ATT with IF SEs |
| `propensity_score_match()` | End-to-end PSM (default logit caliper when `caliper=None`) |

**Default covariates:** `farm_size_ha`, `baseline_yield`, `soil_quality_index`, `historical_rainfall`.

**Matching output columns:** `propensity_score`, `match_pair_id`, `match_role` (`treated` / `control`).

**AIPW** follows Chernozhukov et al. (2018) cross-fitting and Sant'Anna & Zhao (2020) ATT influence functions; doubly robust to misspecification of either nuisance model.

```python
from analysis import aipw_estimator, propensity_score_match, standardized_mean_differences

matched = propensity_score_match(farms_df, k=1)  # 1:1 default; k>1 for k:1 designs
report = standardized_mean_differences(farms_df, matched, covariate_cols=[...])
aipw = aipw_estimator(farms_df, outcome_col="yield_post_intervention", n_folds=5)
```

#### Difference-in-differences — [`did_impact.py`](src/analysis/did_impact.py)

| Function | Description |
|----------|-------------|
| `calculate_did_att()` | Pair-level DiD ATT with paired-bootstrap SE/CI |
| `calculate_avoided_revenue_loss()` | ATT × farm area × cocoa price (optional ATT CI → USD CI) |
| `event_study()` | Leads/lags with unit/time FE; parallel-trends F-test |

**Required panel columns:** `yield_pre_intervention`, `yield_post_intervention`, plus PSM match columns.

**Typical pipeline (PSM + DiD + revenue):**

```python
from analysis import propensity_score_match, calculate_did_att, calculate_avoided_revenue_loss

matched = propensity_score_match(farms_df)
did = calculate_did_att(matched)
revenue = calculate_avoided_revenue_loss(
    did.att,
    matched,
    cocoa_price_usd=3200.0,
    att_ci=(did.ci_low, did.ci_high),
)
```

---

### `src/api` — intervention simulation service

FastAPI app for the **Resilient frontend** to simulate a single-farm intervention prospectively.

| Module | Role |
|--------|------|
| [`main.py`](src/api/main.py) | App factory, lifespan (load model once), routes |
| [`schemas.py`](src/api/schemas.py) | Pydantic request/response models |
| [`config.py`](src/api/config.py) | `APISettings` (env-backed) |
| [`geo_mock.py`](src/api/geo_mock.py) | Deterministic climate/soil mock from lat/lon |
| [`model_loader.py`](src/api/model_loader.py) | Load `YieldSurrogateModel` checkpoint or uninitialized weights |
| [`simulation.py`](src/api/simulation.py) | Counterfactual vs factual MC inference + metrics |
| [`financial.py`](src/api/financial.py) | `avoided_loss_tonnes × cocoa_price_usd` |

#### Run the server

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness (`{"status": "ok"}`) |
| `POST` | `/simulate-intervention` | Avoided loss simulation |
| `POST` | `/simulate-scenario` | Avoided loss under SSP × horizon CMIP6-adjusted ERA5 |

#### `POST /simulate-intervention`

**Request body:**

```json
{
  "farm_location": { "lat": 6.5, "lon": -1.2 },
  "farm_size_ha": 5.0,
  "current_yield": 2.0,
  "intervention_type": "shade_trees",
  "cocoa_price_usd": 3200.0
}
```

| Field | Constraints |
|-------|-------------|
| `farm_location.lat` | −90 … 90 |
| `farm_location.lon` | −180 … 180 |
| `farm_size_ha` | > 0 |
| `current_yield` | ≥ 0 (tonnes/ha) |
| `intervention_type` | `shade_trees`, `agroforestry`, `drought_resistant_variety` |
| `cocoa_price_usd` | ≥ 0 (required) |

**Response (key fields):**

| Field | Units | Meaning |
|-------|-------|---------|
| `baseline_yield_tonnes_per_ha` | t/ha | Counterfactual (no intervention) |
| `projected_yield_tonnes_per_ha` | t/ha | Factual (with intervention) |
| `avoided_loss_tonnes` | tonnes | `max(0, projected − baseline) × farm_size_ha` |
| `financial_impact_usd` | USD | `avoided_loss_tonnes × cocoa_price_usd` |
| `confidence_interval.avoided_loss_tonnes` | tonnes | 90% interval from paired MC samples |

**Simulation steps (v1):**

1. Resolve ERA5 daily climate `[1, 365, 11]` and site static `[1, 10]` from Zarr or live GEE (cached).
2. Build counterfactual vs factual tensors (intervention encoded in static features; shade trees adjust max-temp channel).
3. Run paired Monte Carlo forwards through `YieldSurrogateModel`.
4. Blend MC means with `current_yield` (`YIELD_BLEND_WEIGHT`).
5. Mechanistic intervention registry adjusts temperature/VPD/soil moisture channels where configured.
6. Compute avoided loss, USD impact, and 5th/95th percentile CI on per-sample avoided tonnes.

#### `POST /simulate-scenario`

Runs the same intervention machinery as `/simulate-intervention`, but replaces resolved ERA5 with a **delta-change** daily stack built by [`ScenarioBuilder`](src/counterfactual/cmip6_scenarios.py): monthly CMIP6 anomalies vs historical ERA5 (temperature/humidity additive; precipitation/radiation/wind multiplicative), then recomputed VPD / ET0 / CWD / agronomic indices.

Uses [`CASEJSurrogate`](src/models/casej_surrogate.py) (PINN emulating Asante et al. 2025 CASEJ CO₂ physiology) with explicit SSP CO₂ (ppm) — not the generic `YieldSurrogateModel`. `/simulate-intervention` continues to use `YieldSurrogateModel` until CASEJ passes intervention benchmarks.

Train: `python scripts/generate_casej_training_set.py` → `python scripts/train_casej_surrogate.py` → `models/casej_surrogate.pt` (`CASEJ_CHECKPOINT_PATH`).

**Extra request fields:** `scenario` (`ssp245` \| `ssp585`) and `horizon_year` (`2030` \| `2050` \| `2080`). The horizon selects the CMIP6 climatology window used for those monthly deltas; `CLIMATE_REFERENCE_YEAR` still picks which calendar year is sliced from the adjusted ERA5 timeline for the surrogate.

**Response:** `baseline_yield_tonnes_per_ha` and `projected_yield_tonnes_per_ha` each expose `mean`, `p10`, and `p90` from paired Monte Carlo forwards; `avoided_loss_tonnes` uses per-sample `max(projected − baseline, 0) × farm_size_ha`; `financial_impact_usd_mean` multiplies the **mean** avoided tonnes by `cocoa_price_usd`.

**Conformal UQ (default `CONFORMAL_METHOD=eci_integral`):** parallel CQR on SSP-adjusted climate with online ECI-Integral thresholds per `{scenario}:{horizon}:{region}`. Response includes:

| Field | Description |
|-------|-------------|
| `confidence_interval.method` | e.g. `eci_integral`, `aci`, `cqr` (when `split_cqr`) |
| `confidence_interval.coverage_running_avg` | Rolling mean 90% PI coverage over the last 1000 API updates for this stratum |
| `confidence_interval.avoided_loss_tonnes` | `{ lower, upper, level }` conformal avoided-loss bounds |

Financial impact `ci_low` / `ci_high` use the online conformal interval when conformal is active. Static split-CQR: set `CONFORMAL_METHOD=split_cqr`. See [`docs/conformal_calibration.md`](docs/conformal_calibration.md).

Requires existing directories at `ERA5_ZARR_PATH` and `CMIP6_ZARR_PATH` (see `.env.example`).

---

## Training runbook (GPU / deferred)

Use **[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md)** when you have access to a GPU machine or cluster. It lists:

- Which background training jobs are safe to kill if they never finished on your laptop
- Phases: GEE ingest → exposure checkpoints (AEF, Galileo, AgriFM, ensemble v2 YAML) → yield + CQR → CASEJ scenarios → causal panel eval → `demo_end_to_end.py` / API smoke
- Artifacts to commit or DVC-track in follow-up PRs

---

## End-to-end workflows

### Workflow A — Geospatial plantation mapping

```text
earthengine authenticate
python -m data.gee_auth
python -m data.sentinel_composite --destination local --output data/raw/ghana_composite.tif
# Prepare mask tiles under data/processed/masks/
python -m training.train_prithvi_cocoa --image-dir ... --mask-dir ...
```

### Workflow B — Cohort causal impact (research / evaluation)

```text
# farms_df: farm_id, received_intervention, covariates, yield_pre/post
matched = propensity_score_match(farms_df)
att = calculate_did_att(matched).att
usd = calculate_avoided_revenue_loss(att, matched, cocoa_price_usd=3200.0)
```

### Workflow C — Product intervention preview (frontend)

```text
uvicorn api.main:app --reload
POST /simulate-intervention  →  JSON for UI charts and copy
```

### Workflow D — Future-scenario simulation (CMIP6-adjusted ERA5)

Export or build:

1. Historical ERA5-Land Zarr at `ERA5_ZARR_PATH` (same schema as the API resolver).
2. CMIP6 ensemble Zarr at `CMIP6_ZARR_PATH` with dims including `scenario`, `model`, `time`, and lat/lon — produced by [`cmip6_ingest`](src/data/cmip6_ingest.py) / NASA `NASA/GDDP-CMIP6` workflow.

Then:

```text
POST /simulate-scenario
```

Body matches `/simulate-intervention` plus `scenario` (`ssp245` \| `ssp585`) and `horizon_year` (`2030` \| `2050` \| `2080`). The API applies [`ScenarioBuilder`](src/counterfactual/cmip6_scenarios.py) and returns mean / p10 / p90 yield bands plus avoided tonnes vs the SSP-conditioned baseline.

---

## Testing

```bash
source .venv/bin/activate
pytest                    # full suite
pytest tests/test_api_simulate.py tests/test_api_scenario.py -v
pytest --cov=src          # with coverage (dev extra)
```

| Test file | Covers |
|-----------|--------|
| `test_gee_auth.py` | EE init helpers (mocked where needed) |
| `test_era5_ingest.py` | ERA5 image algebra / export helpers |
| `test_sentinel_composite.py` | S2/S1 composite builders |
| `test_cocoa_dataset.py` | TorchGeo dataset/datamodule |
| `test_prithvi_training.py` | Training script wiring |
| `test_yield_surrogate.py` | Surrogate shapes, MC dropout, uncertainty |
| `test_psm_matching.py` | PSM matching |
| `test_did_impact.py` | DiD ATT + revenue |
| `test_api.py` | `/health` |
| `test_api_simulate.py` | `/simulate-intervention` validation + happy path |
| `test_api_scenario.py` | `/simulate-scenario` with mocked `ScenarioBuilder` + path validation |

**Note:** API tests use `with TestClient(app) as client:` so FastAPI **lifespan** runs and `app.state.yield_model` is initialized.

---

## Data versioning (DVC)

Large files under `data/` and `models/` are **gitignored**. Track them with DVC after configuring a remote:

```bash
dvc add data/raw/your_dataset.tif
git add data/raw/your_dataset.tif.dvc .gitignore
git commit -m "Track dataset with DVC"
```

[`dvc.yaml`](dvc.yaml) currently has empty `stages: {}` — add reproducible stages as pipelines stabilize (GEE export → train → export checkpoint).

---

## Design decisions and limitations

| Topic | Current state | Planned direction |
|-------|---------------|-------------------|
| API geospatial input | `geo_mock` (deterministic fake climate/soil) | Wire `era5_ingest` + site covariates from rasters |
| Yield checkpoint | Optional `MODEL_CHECKPOINT_PATH`; warns if missing | DVC/MLflow-tracked trained weights |
| API vs DiD | API = prospective single-farm; DiD = retrospective cohort ATT | May share yield model outputs as counterfactuals later |
| Segmentation vs yield | Independent stacks today | Joint features from Prithvi tiles → static branch |
| GEE in API | Not called (latency, auth) | Precompute or cache features by tile |
| Auth / CORS | Not implemented | Add for production Resilient deployment |
| DVC pipeline | Empty | Encode export → train → evaluate |

**Intervention uplift registry** in `api/simulation.py` ensures the frontend sees plausible effect sizes during demos; replace or calibrate when a trained surrogate is available.

---

## Contributing

1. Branch from `main`: `feat/your-feature` or `fix/your-fix`.
2. Match existing style (`ruff`, line length 100).
3. Add tests under `tests/` mirroring `src/`.
4. Open a PR against [Resilient-World/cocoa-model](https://github.com/Resilient-World/cocoa-model) with a short summary and test plan.

**Merged capability timeline (PRs #1–#12):** scaffold → GEE auth → ERA5 → Sentinel composite → TorchGeo dataset → Prithvi training → yield surrogate → MC dropout → PSM → DiD → FastAPI intervention API.

---

## License

MIT (see `pyproject.toml`).
