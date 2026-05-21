# Exposure fine-tunes, ensemble v2/v3 — deferred GPU workflow

> **Full avoided-loss stack (yield, CQR, CASEJ, ingest):** see [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).

The **code paths** for AgriFM and TerraMind cocoa fine-tuning, per-region ensemble weights (`ensemble_v2`, opt-in `ensemble_v3`), and production exposure are in this repository. **Artifacts** below are not checked in until you run training on GPU/HPC. Incomplete local runs can be stopped safely; see the runbook section *Local runs that did not finish*.

Production default remains **`ensemble_v2`**. Enable **`ensemble_v3`** via `ENSEMBLE_BACKEND=v3` or `COCOA_EXPOSURE_BACKEND=ensemble_v3` after fitting v3 weights and TerraMind checkpoints.

## When you have GPU (or HPC) access

Run in order from the repo root (with `.venv` activated and Earth Engine authenticated if using GEE-backed labels):

```bash
# Optional: TerraMind / TerraTorch stack (Apache-2.0 TerraMind-1.0-base)
pip install -e ".[terramind]"

# 1. AgriFM pretrained backbone (if missing)
python scripts/download_agrifm_weights.py

# 2. Fine-tune AgriFM on cocoa masks (~50 epochs; MLflow experiment agrifm_cocoa_finetune)
python scripts/train_agrifm_cocoa.py \
  --pretrained models/agrifm/agrifm_s2_pretrained.pt \
  --out models/agrifm_cocoa_seg.pt

# 3. Fine-tune TerraMind on cocoa masks (batch 8, A100-80GB ~6 h/region; MLflow terramind_cocoa_finetune)
python scripts/train_terramind_cocoa.py --out models/terramind_cocoa_seg.pt
# TiM variant (optional): python scripts/train_terramind_cocoa.py --tim --out models/terramind_tim_cocoa_seg.pt

# Smoke test on CPU/CI only (does not replace production checkpoints):
# python scripts/train_agrifm_cocoa.py --synthetic --quick --epochs 1 --max-tiles 100
# python scripts/train_terramind_cocoa.py --synthetic --quick --epochs 1 --max-tiles 100

# 4. Fit per-region ensemble v2 weights (5000 holdout tiles/region; 6/8 F1 gate)
python scripts/fit_ensemble_v2_weights.py

# 5. Fit ensemble v3 weights (NNLS on 5 backends: AEF + Galileo + AgriFM + TerraMind + FDP)
python scripts/fit_ensemble_v3_weights.py

# 6. Benchmarks (Ghana/CIV write benchmark_terramind_<region>_<date>.md)
python scripts/benchmark_backbones.py --backbone terramind --region ghana --quick
python scripts/benchmark_backbones.py --backbone terramind_tim --region civ --quick
```

## TerraMind latency (relative to Galileo-Base)

| Variant | Approx. vs Galileo-B |
|---------|----------------------|
| TerraMind-B | ~1.6× |
| TerraMind-L | ~3.2× |
| TiM (+ LULC/NDVI generation) | +~30% on top of encoder |

Use `terramind_v1_base` for API parity; TiM path adds intermediate token generation before re-encode.

## Artifacts to commit after training (optional follow-up PR)

| Path | Purpose |
|------|---------|
| `models/agrifm_cocoa_seg.pt` | Fine-tuned AgriFM cocoa segmentation |
| `models/terramind_cocoa_seg.pt` | Fine-tuned TerraMind UPerNet head |
| `models/terramind_tim_cocoa_seg.pt` | Optional TiM segmentation variant |
| `config/ensemble_weights.yaml` | Region-specific `ensemble_v2` blend + validation F1 |
| `config/ensemble_weights_v3.yaml` | Region-specific `ensemble_v3` NNLS weights + validation F1 |

Until those files exist locally, ensembles fall back to **default** weights in the YAML files; missing checkpoints log random-init warnings.

## Environment

See `.env.example`:

- `COCOA_EXPOSURE_BACKEND=ensemble_v2` (default) or `ensemble_v3`
- `ENSEMBLE_BACKEND=v2` or `v3`
- `AGRIFM_CHECKPOINT_PATH`, `TERRAMIND_CHECKPOINT_PATH`
- `ENSEMBLE_WEIGHTS_PATH`, `ENSEMBLE_V3_WEIGHTS_PATH`

## CI vs production

- **CI** uses `--synthetic` training and stub YAML defaults; it does not require GPU artifacts.
- **Production** should keep `ensemble_v2` until AgriFM/TerraMind checkpoints and fitted YAML regions are present; v2 fit script should pass the **≥ 6 / 8** regions F1 gate before relying on production blends.
