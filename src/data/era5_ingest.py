"""
Ingest ERA5 daily climate data for Ghana and Côte d'Ivoire via Google Earth Engine.

Computes per-pixel heat-stress day counts (daily max 2 m temperature > 32 °C) and
annual total precipitation for a given year, then exports a multi-band GeoTIFF.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import ee

from data.gee_auth import (
    EarthEngineAuthError,
    EarthEngineNotAuthenticatedError,
    initialize_earth_engine,
)

ERA5_DAILY_COLLECTION = "ECMWF/ERA5/DAILY"
BAND_MAX_TEMP = "maximum_2m_air_temperature"
BAND_PRECIP = "total_precipitation"

# Ghana & Côte d'Ivoire — [west, south, east, north] in degrees
GHANA_CI_BOUNDS: dict[str, float] = {
    "west": -8.6,
    "south": 4.0,
    "east": 1.3,
    "north": 11.2,
}

# Native ERA5 resolution (~0.25°)
ERA5_SCALE_METERS = 27_830
KELVIN_OFFSET = 273.15
HEAT_STRESS_THRESHOLD_C = 32.0

ExportDestination = Literal["drive", "local"]


class Era5ExportError(RuntimeError):
    """Raised when an ERA5 export fails or cannot be started."""


def ghana_ci_geometry(
    bounds: dict[str, float] | None = None,
) -> ee.Geometry:
    """Return an Earth Engine rectangle covering Ghana and Côte d'Ivoire."""
    b = bounds or GHANA_CI_BOUNDS
    return ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])


def daily_max_temp_celsius(image: ee.Image) -> ee.Image:
    """Convert daily maximum 2 m temperature from Kelvin to Celsius."""
    return image.select(BAND_MAX_TEMP).subtract(KELVIN_OFFSET).rename("max_temp_c")


def build_era5_daily_collection(
    year: int,
    roi: ee.Geometry | None = None,
) -> ee.ImageCollection:
    """
    Filter ECMWF/ERA5/DAILY to a calendar year over the region of interest.

    Bands retained: maximum_2m_air_temperature (K), total_precipitation (m).
    """
    region = roi or ghana_ci_geometry()
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    return (
        ee.ImageCollection(ERA5_DAILY_COLLECTION)
        .filterDate(start, end)
        .filterBounds(region)
        .select([BAND_MAX_TEMP, BAND_PRECIP])
    )


def compute_heat_stress_days(
    collection: ee.ImageCollection,
    threshold_c: float = HEAT_STRESS_THRESHOLD_C,
) -> ee.Image:
    """
    Count days per pixel where daily maximum 2 m temperature exceeds threshold_c.

    Returns an Int16 image named ``heat_stress_days``.
    """
    threshold_k = threshold_c + KELVIN_OFFSET

    def _daily_stress(image: ee.Image) -> ee.Image:
        exceeds = image.select(BAND_MAX_TEMP).gt(threshold_k)
        return exceeds.rename("heat_stress").toUint8()

    return (
        collection.map(_daily_stress)
        .sum()
        .rename("heat_stress_days")
        .toInt16()
        .set(
            {
                "heat_stress_threshold_c": threshold_c,
                "description": f"Days with daily max 2m temperature > {threshold_c} °C",
            }
        )
    )


def compute_annual_precipitation(collection: ee.ImageCollection) -> ee.Image:
    """Sum daily total precipitation (m) over the collection period."""
    return (
        collection.select(BAND_PRECIP)
        .sum()
        .rename("annual_precipitation_m")
        .set(
            {
                "units": "m",
                "description": "Annual sum of ERA5 daily total precipitation",
            }
        )
    )


def build_era5_summary_image(
    year: int,
    roi: ee.Geometry | None = None,
    *,
    heat_stress_threshold_c: float = HEAT_STRESS_THRESHOLD_C,
) -> tuple[ee.Image, ee.ImageCollection]:
    """
    Build a multi-band summary image for the given year.

    Bands
    -----
    heat_stress_days : int16
        Count of days with daily max temperature above threshold.
    annual_precipitation_m : float
        Sum of daily precipitation (meters).
    """
    region = roi or ghana_ci_geometry()
    collection = build_era5_daily_collection(year, region)
    heat_stress = compute_heat_stress_days(collection, heat_stress_threshold_c)
    annual_precip = compute_annual_precipitation(collection)

    summary = heat_stress.addBands(annual_precip).clip(region)
    summary = summary.set(
        {
            "year": year,
            "region": "Ghana and Cote d'Ivoire",
            "source": ERA5_DAILY_COLLECTION,
        }
    )
    return summary, collection


def export_to_google_drive(
    image: ee.Image,
    *,
    description: str,
    folder: str,
    region: ee.Geometry,
    scale: int = ERA5_SCALE_METERS,
    max_pixels: int = 10**13,
    wait: bool = True,
    poll_interval_s: int = 30,
) -> ee.batch.Task:
    """
    Start an Earth Engine export task to Google Drive as GeoTIFF.

    Returns the Task object; optionally blocks until completion when wait=True.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()

    if wait:
        _wait_for_task(task, poll_interval_s=poll_interval_s)

    return task


def export_local_geotiff(
    image: ee.Image,
    output_path: str | Path,
    *,
    region: ee.Geometry,
    scale: int = ERA5_SCALE_METERS,
    max_pixels: int = 10**13,
) -> Path:
    """
    Download a GeoTIFF locally.

    Uses geemap when installed; otherwise falls back to ``getDownloadURL``.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import geemap

        geemap.ee_export_image(
            image,
            filename=str(path),
            scale=scale,
            region=region,
            file_per_band=False,
        )
        return path
    except ImportError:
        pass

    params = {
        "image": image,
        "region": region,
        "scale": scale,
        "maxPixels": max_pixels,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    try:
        url = image.getDownloadURL(params)
    except ee.EEException as exc:
        raise Era5ExportError(
            "Local download failed. The region may exceed EE download limits. "
            "Install geemap (`pip install geemap`) or use --export drive.\n"
            f"Original error: {exc}"
        ) from exc

    import urllib.request

    print(f"Downloading GeoTIFF to {path} ...")
    urllib.request.urlretrieve(url, path)
    return path


def _wait_for_task(task: ee.batch.Task, poll_interval_s: int = 30) -> None:
    """Poll an EE batch task until it completes or fails."""
    print(f"Task {task.id} started — waiting for completion ...")
    while True:
        status = task.status()
        state = status.get("state")
        if state == "COMPLETED":
            print(f"Task {task.id} completed.")
            if "destination_uris" in status:
                print(f"  Output: {status['destination_uris']}")
            return
        if state == "FAILED":
            raise Era5ExportError(
                f"Earth Engine export failed: {status.get('error_message', status)}"
            )
        if state == "CANCELLED":
            raise Era5ExportError(f"Earth Engine export cancelled: {task.id}")
        print(f"  Status: {state} — checking again in {poll_interval_s}s")
        time.sleep(poll_interval_s)


def run_era5_export(
    year: int = 2024,
    *,
    export: ExportDestination = "drive",
    output_path: str | Path | None = None,
    drive_folder: str = "resilient_cocoa_model",
    drive_description: str | None = None,
    wait: bool = True,
    project: str | None = None,
) -> ee.batch.Task | Path:
    """
    End-to-end ERA5 ingest: initialize EE, compute summaries, export.

    Parameters
    ----------
    export:
        ``drive`` exports to Google Drive; ``local`` writes a GeoTIFF under output_path.
    output_path:
        Required for local export (default: data/processed/era5_ghana_ci_{year}.tif).
    """
    initialize_earth_engine(project=project)
    roi = ghana_ci_geometry()
    summary, _ = build_era5_summary_image(year, roi)

    description = drive_description or f"era5_ghana_ci_{year}"

    if export == "drive":
        print(
            f"Exporting ERA5 summary ({year}) to Google Drive folder '{drive_folder}' "
            f"as '{description}' ..."
        )
        return export_to_google_drive(
            summary,
            description=description,
            folder=drive_folder,
            region=roi,
            wait=wait,
        )

    dest = output_path or Path("data/processed") / f"era5_ghana_ci_{year}.tif"
    print(f"Exporting ERA5 summary ({year}) locally to {dest} ...")
    return export_local_geotiff(summary, dest, region=roi)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ERA5 daily climate data for Ghana & Côte d'Ivoire and compute "
            "heat-stress days (max 2 m temperature > 32 °C) for a given year."
        )
    )
    parser.add_argument("--year", type=int, default=2024, help="Calendar year (default: 2024)")
    parser.add_argument(
        "--export",
        choices=("drive", "local"),
        default="drive",
        help="Export destination (default: drive)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Local GeoTIFF path when --export local",
    )
    parser.add_argument(
        "--drive-folder",
        default="resilient_cocoa_model",
        help="Google Drive folder for Drive export",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start Drive export and return without waiting for completion",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP project ID (overrides EARTHENGINE_PROJECT)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_era5_export(
            year=args.year,
            export=args.export,
            output_path=args.output,
            drive_folder=args.drive_folder,
            wait=not args.no_wait,
            project=args.project,
        )
    except EarthEngineNotAuthenticatedError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EarthEngineAuthError, Era5ExportError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.export == "drive" and isinstance(result, ee.batch.Task):
        print(f"Drive export task id: {result.id}")
    elif args.export == "local":
        print(f"Saved local GeoTIFF: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
