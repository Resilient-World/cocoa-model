#!/usr/bin/env python3
"""CLI entrypoint for :mod:`training.train_agrifm_cocoa`."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.train_agrifm_cocoa import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
