"""
West Africa CSSV strain regions (Muller et al. 2018).

Simplified polygons for point lookup; replace with full atlas vectors when available.
"""

from __future__ import annotations

import structlog

from functools import lru_cache
from pathlib import Path
from typing import Literal

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

log = structlog.get_logger(__name__)

StrainRegion = Literal["1A", "1B", "1C", "2"]
STRAIN_REGIONS: tuple[StrainRegion, ...] = ("1A", "1B", "1C", "2")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS_PATH = _REPO_ROOT / "data" / "external" / "muller2018_wa_strain_atlas.geojson"


@lru_cache(maxsize=1)
def _load_atlas(path: str) -> tuple[STRtree, list[StrainRegion], list]:
    import json

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Strain atlas not found: {p}")

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    geoms: list = []
    labels: list[StrainRegion] = []
    for feat in data.get("features", []):
        region = str(feat.get("properties", {}).get("strain_region", "2"))
        if region not in STRAIN_REGIONS:
            continue
        geoms.append(shape(feat["geometry"]))
        labels.append(region)  # type: ignore[arg-type]

    if not geoms:
        raise ValueError(f"No strain polygons in {p}")

    return STRtree(geoms), labels, geoms


def lookup_strain_region(
    lat: float,
    lon: float,
    *,
    atlas_path: Path | str | None = None,
    default: StrainRegion = "2",
) -> StrainRegion:
    """
    Point-in-polygon lookup for CSSV strain region at ``(lat, lon)``.

    Returns ``default`` (strain 2) when the point falls outside all atlas polygons.
    """
    path = str(atlas_path or DEFAULT_ATLAS_PATH)
    tree, labels, geoms = _load_atlas(path)
    pt = Point(float(lon), float(lat))
    hits = tree.query(pt)
    if len(hits) == 0:
        log.debug("No strain polygon for (%.4f, %.4f); default=%s", lat, lon, default)
        return default

    for idx in hits:
        if geoms[int(idx)].contains(pt):
            return labels[int(idx)]

    log.debug("Strain atlas miss for (%.4f, %.4f); default=%s", lat, lon, default)
    return default


__all__ = [
    "DEFAULT_ATLAS_PATH",
    "STRAIN_REGIONS",
    "StrainRegion",
    "lookup_strain_region",
]
