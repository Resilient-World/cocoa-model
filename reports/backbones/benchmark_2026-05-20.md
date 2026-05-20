# Cocoa backbone benchmark (2026-05-20)

Held-out spatial tiles over **Côte d'Ivoire + Ghana** with Kalischek et al. (2023) in-situ reference (GEE asset or belt heuristic). **Production backbone: Galileo-Base + seg head** (fine-tuned Galileo-Base; FDP 2025a as weak prior).

Held-out metric leader (untrained run): **FDP-only (2025a prior)** (mIoU=0.864).

| Backbone | mIoU | F1 | Boundary IoU | Latency (ms/tile) | Params (M) |
|----------|------|-----|--------------|-------------------|------------|
| FDP-only (2025a prior) | 0.864 | 0.927 | 0.127 | 0.1 | 0.0 |
| Galileo-Base + seg head | 0.000 | 0.000 | 0.000 | 3701.6 | 0.0 |
| Prithvi-EO-2.0 (6-band proxy) | 0.000 | 0.000 | 0.000 | 1.0 | 0.0 |

> Without ``models/galileo_cocoa_seg.pt``, Galileo mIoU reflects random head weights. Re-run after ``python -m training.train_galileo_cocoa`` for held-out parity.

## Notes

- **FDP-only** uses the 2025a prior thresholded at 0.96 (FDP model card F1-optimal).
- **Galileo-Base** uses :class:`models.galileo_seg.GalileoCocoaSegmentation` (S2×10 + S1 + ERA5 monthly×5 + DEM).
- **Prithvi-EO-2.0** row uses a 6-band proxy stem when TerraTorch checkpoints are not present; swap in ``SemanticSegmentationTask`` for production parity.
- Production exposure API: ``backend='galileo'`` or ``'ensemble'`` in :mod:`data.cocoa_exposure`.
