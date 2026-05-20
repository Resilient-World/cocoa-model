# Cocoa backbone benchmark — AlphaEarth Foundations (2026-05-20)

**Quick evaluation (200 tiles, Galileo-nano, `--quick`).** Re-run overnight with
`python scripts/benchmark_backbones.py --n-tiles 5000` for production parity.

Held-out spatial tiles over **Côte d'Ivoire + Ghana** vs Kalischek et al. (2023) in-situ reference. AlphaEarth Foundations (arXiv:2507.22291) provides pre-computed 64-D annual embeddings on Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) — near-zero inference cost vs ViT backbones.

**Lowest mean error:** AlphaEarth Foundations (AEF) (MAE=0.145). **Production (candidate):** AlphaEarth Foundations + MLP head — pending full GEE benchmark.

| Backbone | Mean error | mIoU | F1 | Boundary IoU | Latency (ms/tile) | Params (M) |
|----------|------------|------|-----|--------------|-------------------|------------|
| AlphaEarth Foundations (AEF) | 0.145 | 0.864 | 0.927 | 0.127 | 0.2 | 0.0 |
| Galileo-Base + seg head | 0.511 | 0.000 | 0.000 | 0.000 | 1187.5 | 1.1 |
| FDP-only (2025a prior) | 0.235 | 0.864 | 0.927 | 0.127 | 0.0 | 0.0 |
| Prithvi-EO-2.0 (6-band proxy) | 0.479 | 0.864 | 0.927 | 0.127 | 0.5 | 0.0 |

## Notes

- **AEF** uses `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64 bands A00–A63) + :class:`models.aef_cocoa_head.AEFCocoaHead`.
- Reported ~23.9% mean error reduction vs other foundation models in DeepMind benchmarks (arXiv:2507.22291).
- **Ensemble exposure** default: `0.5 × AEF + 0.3 × Galileo + 0.2 × FDP`.
- Train AEF head: `python scripts/train_aef_head.py`.
