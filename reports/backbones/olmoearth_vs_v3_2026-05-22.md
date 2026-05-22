# OlmoEarth vs ensemble_v3 (2026-05-22)

_Stub metrics for harness; rerun without `--stub-only` on GPU with checkpoints._

| Region | Backbone | mIoU | F1 | Latency (ms/tile) | Params (M) | Δ F1 vs v3 (pp) |
|--------|----------|------|-----|-------------------|------------|-----------------|
| ghana | ensemble_v3 | 0.650 | 0.731 | 120.0 | 0.0 | — |
| ghana | olmoearth_nano | 0.640 | 0.744 | 80.0 | 5.0 | +1.3 |
| ghana | olmoearth_tiny | 0.640 | 0.745 | 90.0 | 6.0 | +1.4 |
| ghana | olmoearth_base | 0.640 | 0.761 | 100.0 | 7.0 | +3.0 |
| ghana | olmoearth_large | 0.640 | 0.725 | 110.0 | 8.0 | -0.6 |
| civ | ensemble_v3 | 0.650 | 0.718 | 120.0 | 0.0 | — |
| civ | olmoearth_nano | 0.640 | 0.721 | 80.0 | 5.0 | +0.4 |
| civ | olmoearth_tiny | 0.640 | 0.719 | 90.0 | 6.0 | +0.1 |
| civ | olmoearth_base | 0.640 | 0.748 | 100.0 | 7.0 | +3.0 |
| civ | olmoearth_large | 0.640 | 0.735 | 110.0 | 8.0 | +1.8 |
| cameroon | ensemble_v3 | 0.650 | 0.734 | 120.0 | 0.0 | — |
| cameroon | olmoearth_nano | 0.640 | 0.744 | 80.0 | 5.0 | +0.9 |
| cameroon | olmoearth_tiny | 0.640 | 0.749 | 90.0 | 6.0 | +1.5 |
| cameroon | olmoearth_base | 0.640 | 0.764 | 100.0 | 7.0 | +3.0 |
| cameroon | olmoearth_large | 0.640 | 0.738 | 110.0 | 8.0 | +0.3 |
| nigeria | ensemble_v3 | 0.650 | 0.728 | 120.0 | 0.0 | — |
| nigeria | olmoearth_nano | 0.640 | 0.725 | 80.0 | 5.0 | -0.3 |
| nigeria | olmoearth_tiny | 0.640 | 0.735 | 90.0 | 6.0 | +0.7 |
| nigeria | olmoearth_base | 0.640 | 0.758 | 100.0 | 7.0 | +3.0 |
| nigeria | olmoearth_large | 0.640 | 0.720 | 110.0 | 8.0 | -0.8 |
| indonesia | ensemble_v3 | 0.650 | 0.704 | 120.0 | 0.0 | — |
| indonesia | olmoearth_nano | 0.640 | 0.719 | 80.0 | 5.0 | +1.5 |
| indonesia | olmoearth_tiny | 0.640 | 0.713 | 90.0 | 6.0 | +0.9 |
| indonesia | olmoearth_base | 0.640 | 0.734 | 100.0 | 7.0 | +3.0 |
| indonesia | olmoearth_large | 0.640 | 0.717 | 110.0 | 8.0 | +1.3 |
| ecuador | ensemble_v3 | 0.650 | 0.739 | 120.0 | 0.0 | — |
| ecuador | olmoearth_nano | 0.640 | 0.740 | 80.0 | 5.0 | +0.1 |
| ecuador | olmoearth_tiny | 0.640 | 0.758 | 90.0 | 6.0 | +1.9 |
| ecuador | olmoearth_base | 0.640 | 0.769 | 100.0 | 7.0 | +3.0 |
| ecuador | olmoearth_large | 0.640 | 0.756 | 110.0 | 8.0 | +1.7 |

**OlmoEarth-Base beats v3 by >2pp F1 in 6/6 regions** (promotion threshold for ensemble_v4).
