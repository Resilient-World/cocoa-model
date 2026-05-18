# resilient-cocoa-model

Geospatial machine learning for resilient cocoa modeling.

Repository: [Resilient-World/cocoa-model](https://github.com/Resilient-World/cocoa-model)

## Project layout

```
resilient-cocoa-model/
├── data/
│   ├── raw/          # Immutable source inputs
│   ├── interim/      # Intermediate transforms
│   ├── processed/    # Model-ready datasets
│   └── external/     # Third-party reference data
├── notebooks/        # Exploratory analysis
├── src/
│   ├── data/         # Ingestion, ETL, Earth Engine exports
│   ├── models/       # Model definitions (TorchGeo, TerraTorch)
│   ├── training/     # Training loops, MLflow logging
│   └── api/          # FastAPI inference service
├── models/           # Serialized checkpoints (DVC-tracked)
├── tests/
├── scripts/
│   └── init_project.sh
├── pyproject.toml
└── requirements.txt
```

## Open as a Cursor workspace

This project lives at `~/resilient-cocoa-model` (separate from FamilyOS).

- **File → Open Folder…** → `/Users/david/resilient-cocoa-model`, or
- Open `resilient-cocoa-model.code-workspace` for Python/pytest settings preconfigured.

## Quick start

**One-command setup** (Python 3.10+ required; on macOS use Homebrew `python@3.12`):

```bash
cd ~/resilient-cocoa-model
./scripts/setup_venv.sh
source .venv/bin/activate
```

This script recreates `.venv` with Python 3.12, upgrades pip, installs the full geospatial/ML stack (`geopandas`, `torch`, `torchgeo`, `terratorch`, `mlflow`, `dvc`, etc.), verifies imports, and runs tests.

**Project layout only** (git + directories, no pip install):

```bash
chmod +x scripts/init_project.sh
./scripts/init_project.sh
```

**Manual setup** (equivalent to `setup_venv.sh`):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
dvc version   # DVC is installed via pip; repo already has .dvc/ from scaffold
```

## Data versioning

Large artifacts live under `data/` and are ignored by git. Track them with DVC after configuring a remote:

```bash
dvc add data/raw/your_dataset.tif
git add data/raw/your_dataset.tif.dvc .gitignore
git commit -m "Track dataset with DVC"
```

## Environment

Copy `.env.example` to `.env` and set credentials (Earth Engine, cloud storage, MLflow URI). Never commit `.env`.
