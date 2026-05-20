"""
CLI entrypoint for yield surrogate training (DVC ``train_yield`` stage).

Wraps :mod:`scripts.train_yield_surrogate` when LHS parquets and per-farm ERA5
stores exist; otherwise writes a minimal checkpoint for pipeline smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_checkpoint(out: Path) -> None:
    from models.yield_surrogate import YieldSurrogateModel

    out.parent.mkdir(parents=True, exist_ok=True)
    model = YieldSurrogateModel()
    torch.save(model.state_dict(), out)
    logger.warning("Wrote uninitialized yield checkpoint to %s (training data missing)", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train yield surrogate (PINN)")
    parser.add_argument("--climate", type=Path, default=_REPO_ROOT / "data/processed/era5.zarr")
    parser.add_argument("--panel", type=Path, default=_REPO_ROOT / "data/raw/yield_panel.parquet")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models/yield.pt")
    parser.add_argument("--smoke-only", action="store_true", help="Skip training; write minimal checkpoint")
    args = parser.parse_args(argv)

    case2 = _REPO_ROOT / "data/simulations/case2_lhs.parquet"
    almanac = _REPO_ROOT / "data/simulations/almanac_lhs.parquet"
    era5_dir = _REPO_ROOT / "data/era5"
    script = _REPO_ROOT / "scripts/train_yield_surrogate.py"

    if args.smoke_only or not (case2.is_file() and almanac.is_file() and era5_dir.is_dir()):
        if not args.smoke_only:
            logger.warning(
                "LHS parquets or %s missing; writing minimal checkpoint. "
                "Provide data/simulations/*.parquet and data/era5/*.zarr for full training.",
                era5_dir,
            )
        _write_minimal_checkpoint(args.out)
        return 0

    cmd = [
        sys.executable,
        str(script),
        "--case2-parquet",
        str(case2),
        "--almanac-parquet",
        str(almanac),
        "--era5-dir",
        str(era5_dir),
        "--checkpoint",
        str(args.out.with_suffix(".ckpt") if args.out.suffix == ".pt" else args.out),
    ]
    logger.info("Running full yield training: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    if result.returncode != 0:
        return result.returncode

    ckpt = args.out.with_suffix(".ckpt") if args.out.suffix == ".pt" else args.out
    if ckpt.is_file() and args.out.suffix == ".pt":
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        torch.save(state, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
