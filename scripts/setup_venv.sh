#!/usr/bin/env bash
# Create/replace .venv with Python 3.10+ and install the full project stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "${candidate}" &>/dev/null; then
      ver="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      major="${ver%%.*}"
      minor="${ver#*.}"
      if [[ "${major}" -ge 3 && "${minor}" -ge 10 ]]; then
        PYTHON="${candidate}"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Python 3.10+ required. Install python3.12 (e.g. brew install python@3.12) or set PYTHON=python3.12"
  exit 1
fi

echo "==> Using: ${PYTHON} ($(${PYTHON} --version))"
echo "==> Recreating .venv at ${PROJECT_ROOT}/.venv"
rm -rf .venv
"${PYTHON}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing resilient-cocoa-model with all dependencies (this may take several minutes)..."
pip install -e ".[dev]"

echo "==> Verifying key packages..."
python - <<'PY'
import importlib
import sys

packages = [
    "geopandas",
    "rasterio",
    "xarray",
    "netCDF4",
    "ee",
    "torch",
    "torchgeo",
    "terratorch",
    "mlflow",
    "dvc",
    "sklearn",
    "pandas",
    "fastapi",
    "uvicorn",
    "httpx",
]
failed = []
for name in packages:
    try:
        importlib.import_module(name)
        print(f"  OK  {name}")
    except ImportError as e:
        print(f"  FAIL {name}: {e}", file=sys.stderr)
        failed.append(name)
if failed:
    sys.exit(1)
print("All packages imported successfully.")
PY

if [[ ! -d .dvc ]]; then
  echo "==> Initializing DVC..."
  dvc init
else
  echo "==> DVC already initialized"
fi

echo "==> Running tests..."
pytest -q

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
