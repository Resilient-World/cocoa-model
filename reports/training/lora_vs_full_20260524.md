# LoRA vs full fine-tune benchmark

Status: PASS

| Mode | F1 | mIoU | Checkpoint MB | Train seconds |
|---|---:|---:|---:|---:|
| Full fine-tune | 0.9677 | 0.9375 | 1.0715 | 0.013 |
| LoRA adapter | 1.0000 | 1.0000 | 0.0242 | 0.007 |

- Checkpoint reduction: 44.3×
- F1 loss: -3.23 pp
- LoRA trainable parameter fraction: 2.3696%
