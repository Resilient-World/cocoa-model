# AgriFM fine-tune and ensemble v2 — deferred GPU workflow

The **code paths** for AgriFM cocoa fine-tuning, per-region ensemble v2 weights, and production `ensemble_v2` exposure are merged in this repository. The **artifacts** below are intentionally **not** checked in until you run training on a machine with sufficient compute (GPU strongly recommended; full grid search and 50-epoch fine-tune are impractical on CPU).

## When you have GPU (or HPC) access

Run in order from the repo root (with `.venv` activated and Earth Engine authenticated if using GEE-backed labels):

```bash
# 1. AgriFM pretrained backbone (if missing)
python scripts/download_agrifm_weights.py

# 2. Fine-tune AgriFM on cocoa masks (~50 epochs; MLflow experiment agrifm_cocoa_finetune)
python scripts/train_agrifm_cocoa.py \
  --pretrained models/agrifm/agrifm_s2_pretrained.pt \
  --out models/agrifm_cocoa_seg.pt

# Smoke test on CPU/CI only (does not replace production checkpoint):
# python scripts/train_agrifm_cocoa.py --synthetic --quick --epochs 1 --max-tiles 100

# 3. Fit per-region ensemble v2 weights (5000 holdout tiles/region; 6/8 F1 gate)
python scripts/fit_ensemble_v2_weights.py
# Quick dev run: python scripts/fit_ensemble_v2_weights.py --quick

# 4. Optional benchmark report
python scripts/benchmark_backbones.py --write-ensemble-v2-report
```

## Artifacts to commit after training (optional follow-up PR)

| Path | Purpose |
|------|---------|
| `models/agrifm_cocoa_seg.pt` | Fine-tuned AgriFM cocoa segmentation head |
| `config/ensemble_weights.yaml` | Region-specific `ensemble_v2` blend weights + validation F1 |

Until those files exist locally, `ensemble_v2` falls back to **default** weights in `config/ensemble_weights.yaml` and uninitialized or missing AgriFM checkpoints use random-init warnings in logs.

## Environment

See `.env.example`: `COCOA_EXPOSURE_BACKEND=ensemble_v2`, `AGRIFM_CHECKPOINT_PATH`, `ENSEMBLE_WEIGHTS_PATH`.

## CI vs production

- **CI** uses `--synthetic` training and stub YAML defaults; it does not require GPU artifacts.
- **Production** should not enable `ensemble_v2` as the live default until `agrifm_cocoa_seg.pt` and fitted `ensemble_weights.yaml` regions are present and the fit script passes the **≥ 6 / 8** regions F1 gate.
