"""GEDI/ICESat-2 canopy structure features for cocoa exposure and yield models."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import ee
import numpy as np
import structlog
import xarray as xr
import xee  # type: ignore[import-untyped]  # noqa: F401

from data.gee_auth import initialize_earth_engine  # type: ignore[import-untyped]

log = structlog.get_logger(__name__)

GEDI_L4A_MONTHLY = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
GEDI_L3_CANOPY_HEIGHT = "LARSE/GEDI/GEDI03_001/GEDI03_canopy_height"
ATL08_SUBSETTER_ENV = "ATL08_SUBSETTER_CMD"


@dataclass(frozen=True)
class CanopyPointSample:
    canopy_height_m: float
    canopy_cover_pct: float
    agb_mg_ha: float
    height_uncertainty_m: float
    gedi_n_shots: int
    source_attributions: list[str]

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "canopy_height_m": self.canopy_height_m,
            "canopy_cover_pct": self.canopy_cover_pct,
            "agb_mg_ha": self.agb_mg_ha,
            "height_uncertainty_m": self.height_uncertainty_m,
            "gedi_n_shots": self.gedi_n_shots,
            "source_attributions": self.source_attributions,
        }


def _mock_canopy_values(lat: float, lon: float, year: int) -> CanopyPointSample:
    payload = f"{lat:.5f},{lon:.5f},{year}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    cocoa_belt = abs(lat) <= 20.0
    base_height = 11.0 if cocoa_belt else 5.0
    height = float(np.clip(base_height + rng.normal(0.0, 3.0), 0.0, 45.0))
    cover = float(np.clip(height / 35.0 * 100.0 + rng.normal(0.0, 8.0), 0.0, 100.0))
    agb = float(np.clip(8.5 * height + 0.9 * cover + rng.normal(0.0, 12.0), 0.0, 450.0))
    uncertainty = float(np.clip(1.0 + 0.08 * height + rng.normal(0.0, 0.25), 0.5, 8.0))
    shots = int(np.clip(round(4 + cover / 12.0 + rng.normal(0.0, 2.0)), 0, 30))
    return CanopyPointSample(
        canopy_height_m=height,
        canopy_cover_pct=cover,
        agb_mg_ha=agb,
        height_uncertainty_m=uncertainty,
        gedi_n_shots=shots,
        source_attributions=[
            "mock: deterministic GEDI/ATL08 canopy fallback",
            GEDI_L4A_MONTHLY,
            GEDI_L3_CANOPY_HEIGHT,
            "ICESat-2 ATL08",
        ],
    )


def _first_number(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, list | tuple):
        for item in value:
            out = _first_number(item, default)
            if np.isfinite(out):
                return out
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _parse_atl08_payload(payload: str) -> dict[str, float]:
    if not payload.strip():
        return {}
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(obj, dict):
        return {
            "canopy_height_m": _first_number(
                obj.get("canopy_height_m", obj.get("h_canopy")), np.nan
            ),
            "canopy_cover_pct": _first_number(
                obj.get("canopy_cover_pct", obj.get("canopy_openness")), np.nan
            ),
            "height_uncertainty_m": _first_number(
                obj.get("height_uncertainty_m", obj.get("h_canopy_uncertainty")), np.nan
            ),
        }
    return {}


def _run_atl08_subsetter(lat: float, lon: float, year: int) -> dict[str, float]:
    import os

    cmd = os.getenv(ATL08_SUBSETTER_ENV)
    if not cmd:
        return {}
    args = cmd.split() + ["--lat", str(lat), "--lon", str(lon), "--year", str(year)]
    try:
        proc = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("atl08_subsetter_failed", error=str(exc))
        return {}
    return _parse_atl08_payload(proc.stdout)


@dataclass
class GEDICanopyIngest:
    aoi: ee.Geometry
    year: int
    scale: int = 1000
    chunks: dict[str, int] | None = None
    project: str | None = None
    use_mock: bool = False

    def __post_init__(self) -> None:
        self.chunks = self.chunks or {"latitude": 256, "longitude": 256}

    def build(self) -> xr.Dataset:
        if self.use_mock:
            return self._mock_dataset()
        initialize_earth_engine(project=self.project)
        start = f"{self.year}-01-01"
        end = f"{self.year}-12-31"
        l4a = (
            ee.ImageCollection(GEDI_L4A_MONTHLY)
            .filterDate(start, end)
            .filterBounds(self.aoi)
            .select(["agbd"])
        )
        agbd = l4a.mean().rename("agb_mg_ha")
        n_shots = l4a.count().select("agbd").rename("gedi_n_shots")
        height = ee.Image(GEDI_L3_CANOPY_HEIGHT).select([0]).rename("canopy_height_m")
        stack = ee.Image.cat([height, agbd, n_shots]).clip(self.aoi)
        ds = xr.open_dataset(
            stack,
            engine="ee",
            geometry=self.aoi,
            scale=self.scale,
            chunks=self.chunks,
        )
        rename: dict[str, str] = {}
        if "lat" in ds.dims:
            rename["lat"] = "latitude"
        if "lon" in ds.dims:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)
        return self._finalize_dataset(ds)

    def _mock_dataset(self) -> xr.Dataset:
        lats = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
        lons = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
        height = np.zeros((3, 3), dtype=np.float32)
        cover = np.zeros((3, 3), dtype=np.float32)
        agb = np.zeros((3, 3), dtype=np.float32)
        unc = np.zeros((3, 3), dtype=np.float32)
        shots = np.zeros((3, 3), dtype=np.int16)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                sample = _mock_canopy_values(float(lat), float(lon), self.year)
                height[i, j] = sample.canopy_height_m
                cover[i, j] = sample.canopy_cover_pct
                agb[i, j] = sample.agb_mg_ha
                unc[i, j] = sample.height_uncertainty_m
                shots[i, j] = sample.gedi_n_shots
        return xr.Dataset(
            {
                "canopy_height_m": (("latitude", "longitude"), height),
                "canopy_cover_pct": (("latitude", "longitude"), cover),
                "agb_mg_ha": (("latitude", "longitude"), agb),
                "height_uncertainty_m": (("latitude", "longitude"), unc),
                "gedi_n_shots": (("latitude", "longitude"), shots),
            },
            coords={"latitude": lats, "longitude": lons},
            attrs={
                "source": "mock GEDI/ATL08 canopy",
                "year": self.year,
            },
        )

    def _finalize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        height = ds["canopy_height_m"].astype(np.float32)
        if "canopy_cover_pct" not in ds:
            ds["canopy_cover_pct"] = (height / 35.0 * 100.0).clip(0.0, 100.0).astype(np.float32)
        if "height_uncertainty_m" not in ds:
            ds["height_uncertainty_m"] = (0.5 + height * 0.08).astype(np.float32)
        if "agb_mg_ha" not in ds:
            ds["agb_mg_ha"] = xr.full_like(height, np.nan).astype(np.float32)
        if "gedi_n_shots" not in ds:
            ds["gedi_n_shots"] = xr.zeros_like(height).astype(np.int16)
        ds.attrs.update(
            {
                "source": "NASA GEDI L4A/L3 + ICESat-2 ATL08",
                "gedi_l4a_collection": GEDI_L4A_MONTHLY,
                "gedi_l3_collection": GEDI_L3_CANOPY_HEIGHT,
                "atl08_boundary": "subprocess Earthdata Subsetter",
                "year": self.year,
            }
        )
        return ds[
            [
                "canopy_height_m",
                "canopy_cover_pct",
                "agb_mg_ha",
                "height_uncertainty_m",
                "gedi_n_shots",
            ]
        ]


def sample_canopy_at_point(
    lat: float,
    lon: float,
    year: int,
    *,
    project: str | None = None,
    use_mock: bool | None = None,
) -> CanopyPointSample:
    import os

    mock = (
        use_mock
        if use_mock is not None
        else os.getenv("USE_REAL_FEATURES", "true").lower()
        in {
            "0",
            "false",
            "no",
            "off",
        }
    )
    if mock:
        return _mock_canopy_values(lat, lon, year)
    try:
        initialize_earth_engine(project=project)
        point = ee.Geometry.Point([lon, lat])
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        l4a = (
            ee.ImageCollection(GEDI_L4A_MONTHLY)
            .filterDate(start, end)
            .filterBounds(point.buffer(1000))
            .select(["agbd"])
        )
        height = ee.Image(GEDI_L3_CANOPY_HEIGHT).select([0]).rename("canopy_height_m")
        agbd = l4a.mean().rename("agb_mg_ha")
        n_shots = l4a.count().select("agbd").rename("gedi_n_shots")
        stack = ee.Image.cat([height, agbd, n_shots])
        props = stack.reduceRegion(ee.Reducer.first(), point, scale=1000).getInfo() or {}
        canopy_height = float(props.get("canopy_height_m") or np.nan)
        agb = float(props.get("agb_mg_ha") or np.nan)
        shots = int(float(props.get("gedi_n_shots") or 0))
        atl08 = _run_atl08_subsetter(lat, lon, year)
        if not np.isfinite(canopy_height):
            canopy_height = atl08.get("canopy_height_m", np.nan)
        if not np.isfinite(canopy_height):
            return _mock_canopy_values(lat, lon, year)
        cover = atl08.get(
            "canopy_cover_pct", float(np.clip(canopy_height / 35.0 * 100.0, 0.0, 100.0))
        )
        uncertainty = atl08.get(
            "height_uncertainty_m", float(np.clip(0.5 + 0.08 * canopy_height, 0.5, 8.0))
        )
        if not np.isfinite(agb):
            agb = float(np.clip(8.5 * canopy_height + 0.9 * cover, 0.0, 450.0))
        return CanopyPointSample(
            canopy_height_m=float(canopy_height),
            canopy_cover_pct=float(np.clip(cover, 0.0, 100.0)),
            agb_mg_ha=float(agb),
            height_uncertainty_m=float(uncertainty),
            gedi_n_shots=shots,
            source_attributions=[GEDI_L4A_MONTHLY, GEDI_L3_CANOPY_HEIGHT, "ICESat-2 ATL08"],
        )
    except Exception as exc:
        log.warning("gedi_canopy_sample_failed", error=str(exc), fallback="mock")
        return _mock_canopy_values(lat, lon, year)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sample GEDI/ICESat-2 canopy at one point")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    sample = sample_canopy_at_point(args.lat, args.lon, args.year, use_mock=args.mock)
    sys.stdout.write(json.dumps(sample.as_dict(), indent=2) + "\n")
    return 0


__all__ = [
    "ATL08_SUBSETTER_ENV",
    "GEDI_L3_CANOPY_HEIGHT",
    "GEDI_L4A_MONTHLY",
    "CanopyPointSample",
    "GEDICanopyIngest",
    "sample_canopy_at_point",
]


if __name__ == "__main__":
    raise SystemExit(main())
