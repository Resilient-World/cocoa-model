# Galileo Foundation Model Migration

We replace Prithvi-EO-2.0 with Galileo (Tseng et al. 2025, MIT licensed) as the foundation
backbone for cocoa exposure mapping and yield-loss regression.

## Why

- Galileo-Base ranks 3.0 on image tasks and 1.8 on pixel time series vs Prithvi-2.0 at 11.7 on
  images (NASA Harvest benchmark).
- Native support for our full stack: S1, S2, ERA5, TerraClimate, SRTM, DynamicWorld, VIIRS,
  location.
- On m-Cashew-Plant (closest perennial-tree-crop analog), Galileo-Base achieves 33.0% mIoU at
  100% training data; 30.2% at 1%.
- MIT license; commercial-safe.

## How

- `src/models/vendor/single_file_galileo.py` — vendored encoder (MIT).
- `src/models/galileo_loader.py` — HuggingFace weight loader.
- `src/models/galileo_features.py` — modality-aware feature extractor.
- `src/models/cocoa_head.py` — linear and MLP heads for downstream tasks.

## Recommended config

- Size: ViT-Base for production, ViT-Nano for CPU dev.
- Patch size: 8 (90% of patch=2 accuracy at ~12% of FLOPs).
- Probe: linear/MLP on frozen features for v1; LoRA finetune for v2 once >1000 labeled cocoa
  parcels are available.

## Reference

Tseng et al., "Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models,"
ICML 2025. arXiv:2502.09356. Code: github.com/nasaharvest/galileo
