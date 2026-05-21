#!/usr/bin/env python3
"""
Download AgriFM Sentinel-2 pretrained weights and verify integrity.

Official mirrors (AgriFM README):
- OneDrive: https://hkuhk-my.sharepoint.com/.../AgriFM weights
- GLASS: https://glass.hku.hk/casual/AgriFM/

On first successful download a sidecar ``.sha256`` file is written; subsequent
runs verify against that digest.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = _REPO_ROOT / "models" / "agrifm"
DEFAULT_OUT_FILE = DEFAULT_OUT_DIR / "agrifm_s2_pretrained.pt"
DEFAULT_SHA_FILE = DEFAULT_OUT_FILE.with_suffix(DEFAULT_OUT_FILE.suffix + ".sha256")

# GLASS hosts a direct casual download page; users may need manual download from OneDrive.
GLASS_WEIGHTS_PAGE = "https://glass.hku.hk/casual/AgriFM/"
# Placeholder: populate when a stable direct URL is published; None skips auto-download.
MODEL_HASHES: dict[str, str | None] = {
    "agrifm_s2_pretrained.pt": None,
}

DOWNLOAD_URLS: list[str] = [
    # Add direct artifact URLs here when available from GLASS/OneDrive mirrors.
]


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_sha256_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}\n", encoding="utf-8")
    logging.info("Wrote SHA256 sidecar %s", sidecar)


def _read_sha256_sidecar(path: Path) -> str | None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return None
    return sidecar.read_text(encoding="utf-8").strip().split()[0]


def _verify(path: Path, expected: str | None) -> None:
    digest = _sha256_file(path)
    sidecar = _read_sha256_sidecar(path)
    if sidecar is None:
        _write_sha256_sidecar(path, digest)
        logging.info("Recorded SHA256 for %s: %s", path.name, digest)
        MODEL_HASHES[path.name] = digest
        return
    if digest != sidecar:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {sidecar}, got {digest}")
    if expected is not None and digest != expected:
        raise RuntimeError(f"SHA256 mismatch vs MODEL_HASHES for {path.name}")
    logging.info("SHA256 verified for %s", path.name)


def _download_url(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def download_weights(
    dest: Path = DEFAULT_OUT_FILE,
    *,
    force: bool = False,
) -> Path:
    """Download or verify AgriFM S2 weights at ``dest``."""
    if dest.is_file() and not force:
        _verify(dest, MODEL_HASHES.get(dest.name))
        return dest

    last_error: Exception | None = None
    for url in DOWNLOAD_URLS:
        try:
            _download_url(url, dest)
            _verify(dest, MODEL_HASHES.get(dest.name))
            return dest
        except Exception as exc:
            last_error = exc
            logging.warning("Download failed from %s: %s", url, exc)

    msg = (
        f"Automatic download unavailable. Place AgriFM S2 weights at:\n  {dest}\n"
        f"Mirrors: {GLASS_WEIGHTS_PAGE}\n"
        "See AgriFM README OneDrive link for pretrained AgriFM.pth (rename to "
        f"{dest.name})."
    )
    if last_error is not None:
        msg += f"\nLast error: {last_error}"
    raise FileNotFoundError(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download AgriFM S2 pretrained weights")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        path = download_weights(args.out, force=args.force)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    logging.info("AgriFM weights ready at %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
