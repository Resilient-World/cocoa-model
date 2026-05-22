#!/usr/bin/env python3
"""Download OlmoEarth HF checkpoints (nano/tiny/base/large)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.backbones.olmoearth_backbone import HF_REPO_BY_SIZE

DEFAULT_OUT = _REPO_ROOT / "models" / "olmoearth"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(HF_REPO_BY_SIZE), nargs="+", default=["base"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logging.error("Install olmoearth extra: pip install -e '.[olmoearth]'")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        repo = HF_REPO_BY_SIZE[size]
        logging.info("Downloading %s → %s", repo, args.out_dir / size)
        local = snapshot_download(repo_id=repo, cache_dir=str(args.out_dir / size))
        marker = args.out_dir / f"{size}.downloaded"
        marker.write_text(f"{local}\n", encoding="utf-8")
        logging.info("Cached at %s", local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
