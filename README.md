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

```bash
cd ~/resilient-cocoa-model
chmod +x scripts/init_project.sh
./scripts/init_project.sh   # safe to re-run; skips existing git/DVC

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
dvc init                      # after pip install, if not already done
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
