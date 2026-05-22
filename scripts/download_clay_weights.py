#!/usr/bin/env python3
"""Download Clay v1.5 weights from HuggingFace."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.backbones.clay_backbone import CLAY_HF_REPO  # noqa: E402

DEFAULT_OUT = _REPO_ROOT / "models" / "clay"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logging.error("Install clay extra: pip install -e '.[clay]'")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=CLAY_HF_REPO, cache_dir=str(args.out_dir))
    (args.out_dir / "v1.5.downloaded").write_text(f"{local}\n", encoding="utf-8")
    logging.info("Clay weights cached at %s", local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
