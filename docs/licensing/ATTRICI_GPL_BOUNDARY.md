# ATTRICI GPLv3 Boundary

ATTRICI (https://github.com/ISI-MIP/attrici, GPLv3) is invoked as an **external subprocess**
that reads/writes NetCDF files. The Resilient Cocoa codebase never imports any `attrici.*`
module directly. This preserves the "mere aggregation" boundary under GPLv3 §5 and keeps
the rest of the codebase under our chosen commercial license.

Implementation rules (enforced by CI grep):
- NO `import attrici` anywhere in src/
- NO `from attrici` anywhere in src/
- ATTRICI is installed in a separate venv (`.venv-attrici/`)
- All interaction goes through subprocess.run() with NetCDF I/O
- Grid-cell detrend orchestration: `src/counterfactual/attrici_runner.py`
- Full-grid factual NetCDF runs: `scripts/run_attrici_subprocess.py` (isolated `.venv-attrici`)

Reference: Mengel et al. 2021, Geosci. Model Dev., 14, 5269–5284,
https://doi.org/10.5194/gmd-14-5269-2021
