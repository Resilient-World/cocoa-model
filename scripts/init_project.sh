#!/usr/bin/env bash
# Initialize git, DVC, and standard directory layout for resilient-cocoa-model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "==> Project root: ${PROJECT_ROOT}"

# Directory structure
DIRS=(
  "data/raw"
  "data/interim"
  "data/processed"
  "data/external"
  "notebooks"
  "src/data"
  "src/models"
  "src/api"
  "src/training"
  "tests"
  "models"
  "reports/figures"
  "scripts"
)

echo "==> Creating directories..."
for dir in "${DIRS[@]}"; do
  mkdir -p "${dir}"
done

# Placeholder files so empty dirs are tracked (data/ itself is gitignored except .gitkeep)
touch data/raw/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch notebooks/.gitkeep
touch models/.gitkeep
touch reports/figures/.gitkeep

# Python package markers
for pkg in src src/data src/models src/api src/training; do
  touch "${pkg}/__init__.py"
done
touch tests/__init__.py

# Git
if [[ -d .git ]]; then
  echo "==> Git repository already exists; skipping git init"
else
  echo "==> Initializing git repository..."
  git init
  git branch -M main 2>/dev/null || true
fi

# DVC
if [[ -d .dvc ]]; then
  echo "==> DVC already initialized; skipping dvc init"
else
  if command -v dvc &>/dev/null; then
    echo "==> Initializing DVC..."
    dvc init
  else
    echo "WARNING: 'dvc' not found on PATH. Run 'pip install dvc' then 'dvc init' manually."
  fi
fi

# Optional: wire default remote (uncomment and set URL after creating remote storage)
# dvc remote add -d storage s3://your-bucket/dvc-store 2>/dev/null || true

echo ""
echo "Done. Next steps:"
echo "  1. python -m venv .venv && source .venv/bin/activate"
echo "  2. python -m pip install --upgrade pip setuptools wheel"
echo "  3. pip install -e '.[dev]'   # or: pip install -r requirements.txt"
echo "  4. cp .env.example .env      # add secrets locally (never commit .env)"
echo "  5. dvc remote add -d <name> <url>   # optional remote for data versioning"
echo "  6. git add . && git commit -m 'Initial project scaffold'"
