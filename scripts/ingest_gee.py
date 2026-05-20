#!/usr/bin/env python3
"""
GEE ingest stage marker: ensure ``data/raw/`` AOI assets exist for DVC downstream stages.

Writes ``data/raw/ingest_manifest.json`` documenting AOI paths and optional ERA5 export
targets. Full raster export is handled by ``data.era5_ingest`` / ``data.sentinel_composite``.

Example::

    python scripts/ingest_gee.py
    python scripts/ingest_gee.py --write-era5-stub
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_AOI = RAW_DIR / "cocoa_aoi.geojson"
MANIFEST = RAW_DIR / "ingest_manifest.json"


def _default_aoi_geojson() -> dict:
    """Minimal West Africa cocoa belt AOI if file missing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "cocoa_belt_wa"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-9.0, 4.0],
                            [2.0, 4.0],
                            [2.0, 11.0],
                            [-9.0, 11.0],
                            [-9.0, 4.0],
                        ]
                    ],
                },
            }
        ],
    }


def run_ingest(*, write_era5_stub: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_AOI.is_file():
        DEFAULT_AOI.write_text(json.dumps(_default_aoi_geojson(), indent=2), encoding="utf-8")
        logging.info("Wrote default AOI → %s", DEFAULT_AOI)

    manifest = {
        "date": date.today().isoformat(),
        "aoi_geojson": str(DEFAULT_AOI.relative_to(_REPO_ROOT)),
        "era5_target_zarr": "data/processed/era5_2020_2024.zarr",
        "sentinel_target": "data/processed/s2_s1.tif",
        "notes": "Run era5_ingest / sentinel_composite for full GEE exports.",
    }
    if write_era5_stub:
        stub = _REPO_ROOT / "data" / "processed" / "era5_ingest_requested.json"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(json.dumps({"requested": True, "aoi": str(DEFAULT_AOI)}), encoding="utf-8")
        manifest["era5_stub"] = str(stub.relative_to(_REPO_ROOT))

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote ingest manifest → %s", MANIFEST)
    return MANIFEST


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GEE ingest stage (AOI + manifest)")
    parser.add_argument("--write-era5-stub", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ingest(write_era5_stub=args.write_era5_stub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
