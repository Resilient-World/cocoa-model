"""
Build a cloud-free Sentinel-2 / Sentinel-1 composite for Ghana (dry season).

Creates a median Sentinel-2 SR composite with QA60 cloud masking, NDVI/EVI
indices, merges with median Sentinel-1 GRD VV/VH backscatter, and exports a
multi-band GeoTIFF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import ee

from data.era5_ingest import (
    Era5ExportError,
    export_local_geotiff,
    export_to_google_drive,
)
from data.gee_auth import (
    EarthEngineAuthError,
    EarthEngineNotAuthenticatedError,
    initialize_earth_engine,
)

S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"

# Ghana — [west, south, east, north]
GHANA_BOUNDS: dict[str, float] = {
    "west": -3.25,
    "south": 4.7,
    "east": 1.2,
    "north": 11.2,
}

# Dry season: December 2023 – March 2024 (end date exclusive)
DRY_SEASON_START = "2023-12-01"
DRY_SEASON_END = "2024-04-01"

S2_OPTICAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_REFLECTANCE_SCALE = 10_000

# Export at 10 m (S2 native); S1 IW GRD ~10 m
COMPOSITE_EXPORT_SCALE_M = 10

ExportDestination = Literal["drive", "local"]

# QA60 bit offsets (see Sentinel-2 user handbook)
QA60_CLOUD_BIT = 1 << 10
QA60_CIRRUS_BIT = 1 << 11


class SentinelCompositeError(RuntimeError):
    """Raised when composite construction or export fails."""


def ghana_geometry(bounds: dict[str, float] | None = None) -> ee.Geometry:
    """Return an Earth Engine rectangle covering Ghana."""
    b = bounds or GHANA_BOUNDS
    return ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])


def mask_s2_clouds_qa60(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cirrus using the Sentinel-2 QA60 band.

    Clears bit 10 (opaque clouds) and bit 11 (cirrus) per the standard
    Copernicus S2 cloud-masking recipe.
    """
    qa = image.select("QA60")
    cloud_clear = qa.bitwiseAnd(QA60_CLOUD_BIT).eq(0)
    cirrus_clear = qa.bitwiseAnd(QA60_CIRRUS_BIT).eq(0)
    return image.updateMask(cloud_clear.And(cirrus_clear))


def scale_s2_surface_reflectance(image: ee.Image) -> ee.Image:
    """Convert DN to surface reflectance (0–1 range)."""
    optical = image.select(S2_OPTICAL_BANDS).divide(S2_REFLECTANCE_SCALE)
    return image.addBands(optical, overwrite=True)


def prepare_s2_image(image: ee.Image) -> ee.Image:
    """Apply QA60 cloud mask and surface-reflectance scaling."""
    masked = mask_s2_clouds_qa60(image)
    scaled = scale_s2_surface_reflectance(masked)
    return scaled.copyProperties(image, ["system:time_start"])


def build_s2_collection(
    roi: ee.Geometry,
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
) -> ee.ImageCollection:
    """Load cloud-masked Sentinel-2 SR for the dry-season window."""
    return (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(prepare_s2_image)
    )


def build_s2_median_composite(
    roi: ee.Geometry | None = None,
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
) -> ee.Image:
    """Median composite of cloud-masked Sentinel-2 surface reflectance."""
    region = roi or ghana_geometry()
    collection = build_s2_collection(region, start_date, end_date)
    median = collection.median().select(S2_OPTICAL_BANDS).clip(region)
    return median.set(
        {
            "composite_method": "median",
            "season": "dry",
            "start_date": start_date,
            "end_date": end_date,
            "source": S2_SR_COLLECTION,
        }
    )


def compute_ndvi(image: ee.Image) -> ee.Image:
    """Normalized Difference Vegetation Index from B8 (NIR) and B4 (red)."""
    return image.addBands(
        image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    )


def compute_evi(image: ee.Image) -> ee.Image:
    """
    Enhanced Vegetation Index.

    EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))
    """
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2"),
        },
    ).rename("EVI")
    return image.addBands(evi)


def add_spectral_indices(image: ee.Image) -> ee.Image:
    """Append NDVI and EVI bands to a Sentinel-2 reflectance image."""
    return compute_evi(compute_ndvi(image))


def build_s1_collection(
    roi: ee.Geometry,
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
) -> ee.ImageCollection:
    """Filter Sentinel-1 GRD IW scenes with VV and VH polarizations."""
    return (
        ee.ImageCollection(S1_GRD_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def build_s1_median_backscatter(
    roi: ee.Geometry | None = None,
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
) -> ee.Image:
    """Median VV/VH backscatter (dB) composite for the dry season."""
    region = roi or ghana_geometry()
    collection = build_s1_collection(region, start_date, end_date)
    median = collection.median().rename(["S1_VV", "S1_VH"]).clip(region)
    return median.set(
        {
            "composite_method": "median",
            "season": "dry",
            "start_date": start_date,
            "end_date": end_date,
            "source": S1_GRD_COLLECTION,
            "units": "dB",
        }
    )


def combine_optical_sar(
    s2_image: ee.Image,
    s1_image: ee.Image,
) -> ee.Image:
    """Merge Sentinel-2 optical + indices with Sentinel-1 SAR bands."""
    s2_bands = s2_image.bandNames()
    combined = s2_image.addBands(s1_image)
    return combined.set(
        {
            "description": (
                "Sentinel-2 median SR (QA60 masked) with NDVI/EVI + "
                "Sentinel-1 median VV/VH"
            ),
            "region": "Ghana",
            "s2_bands": s2_bands,
            "s1_bands": s1_image.bandNames(),
        }
    )


def build_ghana_dry_season_composite(
    roi: ee.Geometry | None = None,
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
) -> ee.Image:
    """
    Full pipeline: S2 median + indices, S1 median, merged multi-band image.

    Band order: B2–B12 (selected), NDVI, EVI, S1_VV, S1_VH.
    """
    region = roi or ghana_geometry()
    s2_median = build_s2_median_composite(region, start_date, end_date)
    s2_with_indices = add_spectral_indices(s2_median)
    s1_median = build_s1_median_backscatter(region, start_date, end_date)
    return combine_optical_sar(s2_with_indices, s1_median)


def export_composite(
    image: ee.Image,
    *,
    export: ExportDestination = "drive",
    description: str = "ghana_s2_s1_dry_2023_2024",
    drive_folder: str = "resilient_cocoa_model",
    output_path: str | Path | None = None,
    region: ee.Geometry | None = None,
    scale: int = COMPOSITE_EXPORT_SCALE_M,
    wait: bool = True,
) -> ee.batch.Task | Path:
    """Export the composite to Google Drive or a local GeoTIFF."""
    roi = region or ghana_geometry()

    if export == "drive":
        print(f"Exporting composite to Google Drive ({description}) ...")
        try:
            return export_to_google_drive(
                image,
                description=description,
                folder=drive_folder,
                region=roi,
                scale=scale,
                wait=wait,
            )
        except Era5ExportError as exc:
            raise SentinelCompositeError(str(exc)) from exc

    dest = output_path or Path("data/processed") / f"{description}.tif"
    print(f"Exporting composite locally to {dest} ...")
    try:
        return export_local_geotiff(image, dest, region=roi, scale=scale)
    except Era5ExportError as exc:
        raise SentinelCompositeError(str(exc)) from exc


def run_sentinel_export(
    *,
    export: ExportDestination = "drive",
    start_date: str = DRY_SEASON_START,
    end_date: str = DRY_SEASON_END,
    output_path: str | Path | None = None,
    drive_folder: str = "resilient_cocoa_model",
    drive_description: str | None = None,
    wait: bool = True,
    project: str | None = None,
) -> ee.batch.Task | Path:
    """Initialize Earth Engine, build composite, and export."""
    initialize_earth_engine(project=project)
    roi = ghana_geometry()
    composite = build_ghana_dry_season_composite(roi, start_date, end_date)
    description = drive_description or "ghana_s2_s1_dry_2023_2024"
    return export_composite(
        composite,
        export=export,
        description=description,
        drive_folder=drive_folder,
        output_path=output_path,
        region=roi,
        wait=wait,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cloud-free Sentinel-2 median composite (QA60) with NDVI/EVI plus "
            "Sentinel-1 VV/VH median backscatter for Ghana, dry season Dec 2023–Mar 2024."
        )
    )
    parser.add_argument(
        "--start",
        default=DRY_SEASON_START,
        help=f"Start date inclusive (default: {DRY_SEASON_START})",
    )
    parser.add_argument(
        "--end",
        default=DRY_SEASON_END,
        help=f"End date exclusive (default: {DRY_SEASON_END})",
    )
    parser.add_argument(
        "--export",
        choices=("drive", "local"),
        default="drive",
        help="Export destination (default: drive)",
    )
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=None,
        dest="output",
        help="Local GeoTIFF path when --export local",
    )
    parser.add_argument(
        "--drive-folder",
        default="resilient_cocoa_model",
        help="Google Drive folder",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Export task / file name",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for Drive export completion",
    )
    parser.add_argument("--project", default=None, help="GCP project ID")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is not None and args.export != "local":
        args.export = "local"
    try:
        result = run_sentinel_export(
            export=args.export,
            start_date=args.start,
            end_date=args.end,
            output_path=args.output,
            drive_folder=args.drive_folder,
            drive_description=args.description,
            wait=not args.no_wait,
            project=args.project,
        )
    except EarthEngineNotAuthenticatedError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EarthEngineAuthError, SentinelCompositeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.export == "drive" and isinstance(result, ee.batch.Task):
        print(f"Drive export task id: {result.id}")
    elif args.export == "local":
        print(f"Saved local GeoTIFF: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
