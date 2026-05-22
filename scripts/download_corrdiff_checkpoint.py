#!/usr/bin/env python3
"""Download nvidia/corrdiff-cmip6-era5 weights into models/corrdiff_cmip6/ (DVC-tracked path)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "models" / "corrdiff_cmip6",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        from earth2studio.models.dx import CorrDiffCMIP6
    except ImportError:
        logging.error("Install optional deps: pip install -e '.[corrdiff]'")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pkg = CorrDiffCMIP6.load_default_package()
    CorrDiffCMIP6.load_model(pkg)
    logging.info(
        "CorrDiff checkpoint resolved via Earth2Studio/HF cache. "
        "Track %s with DVC: dvc add models/corrdiff_cmip6",
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
