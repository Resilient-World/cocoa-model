"""Authenticate and initialize the Google Earth Engine Python API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import ee

# Kumasi, Ghana — representative cocoa-belt test location (lon, lat)
GHANA_TEST_POINT: tuple[float, float] = (-1.6244, 6.6885)
SRTM_ELEVATION_IMAGE = "USGS/SRTMGL1_003"
SRTM_ELEVATION_BAND = "elevation"

AUTHENTICATE_INSTRUCTIONS = """
Google Earth Engine is not authenticated.

1. Activate your project virtualenv (if used):
     source .venv/bin/activate

2. Run the Earth Engine authentication flow:
     earthengine authenticate

3. Set your Google Cloud project ID (Earth Engine enabled):
     export EARTHENGINE_PROJECT=your-gcp-project-id

   Or add EARTHENGINE_PROJECT to your .env file in the project root.

4. Retry this script:
     python -m data.gee_auth

Service account (CI / headless) instead of user credentials:
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
     export EARTHENGINE_PROJECT=your-gcp-project-id
""".strip()


class EarthEngineAuthError(RuntimeError):
    """Raised when Earth Engine cannot be authenticated or initialized."""


class EarthEngineNotAuthenticatedError(EarthEngineAuthError):
    """Raised when no usable Earth Engine credentials are available."""


def _load_dotenv() -> None:
    """Load .env from project root when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


def _resolve_project(project: str | None) -> str | None:
    return project or os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _is_auth_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "not authenticated",
        "authenticate",
        "credentials",
        "please authorize",
        "no project",
        "project is required",
        "invalid_grant",
        "could not find default credentials",
        "application default credentials",
    )
    return any(marker in message for marker in markers)


def _is_already_initialized_error(exc: BaseException) -> bool:
    return "already initialized" in str(exc).lower()


def _initialize_with_service_account(project: str | None) -> None:
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise EarthEngineNotAuthenticatedError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set for service-account auth."
        )

    path = Path(key_path).expanduser()
    if not path.is_file():
        raise EarthEngineAuthError(f"Service account key file not found: {path}")

    email = os.environ.get("EARTHENGINE_SERVICE_ACCOUNT")
    if not email:
        raise EarthEngineAuthError(
            "Set EARTHENGINE_SERVICE_ACCOUNT to the service account email "
            "when using GOOGLE_APPLICATION_CREDENTIALS."
        )

    credentials = ee.ServiceAccountCredentials(email, str(path))
    ee.Initialize(credentials, project=project)


def initialize_earth_engine(
    project: str | None = None,
    *,
    opt_url: str | None = None,
) -> None:
    """
    Initialize the Earth Engine API using existing credentials.

    Uses a service account when GOOGLE_APPLICATION_CREDENTIALS is set;
    otherwise expects credentials from ``earthengine authenticate``.

    Parameters
    ----------
    project:
        Google Cloud project ID with Earth Engine enabled. Falls back to
        EARTHENGINE_PROJECT or GOOGLE_CLOUD_PROJECT environment variables.
    opt_url:
        Optional Earth Engine API base URL (rarely needed).

    Raises
    ------
    EarthEngineNotAuthenticatedError
        If the user has not authenticated and no service account is configured.
    EarthEngineAuthError
        For other initialization failures (invalid project, API disabled, etc.).
    """
    _load_dotenv()
    resolved_project = _resolve_project(project)

    try:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            _initialize_with_service_account(resolved_project)
        else:
            if opt_url:
                ee.Initialize(project=resolved_project, opt_url=opt_url)
            else:
                ee.Initialize(project=resolved_project)
    except ee.EEException as exc:
        if _is_already_initialized_error(exc):
            return
        if _is_auth_error(exc):
            raise EarthEngineNotAuthenticatedError(
                f"{exc}\n\n{AUTHENTICATE_INSTRUCTIONS}"
            ) from exc
        raise EarthEngineAuthError(
            f"Failed to initialize Earth Engine: {exc}\n\n"
            "Check that Earth Engine is enabled for your GCP project and that "
            "EARTHENGINE_PROJECT is set correctly."
        ) from exc
    except Exception as exc:
        if _is_auth_error(exc):
            raise EarthEngineNotAuthenticatedError(
                f"{exc}\n\n{AUTHENTICATE_INSTRUCTIONS}"
            ) from exc
        raise EarthEngineAuthError(f"Failed to initialize Earth Engine: {exc}") from exc


def get_elevation_at_point(
    lon: float,
    lat: float,
    *,
    project: str | None = None,
) -> float:
    """
    Sample SRTM elevation (meters) at a single lon/lat point.

    Initializes Earth Engine if it has not been initialized yet in this process.
    """
    initialize_earth_engine(project=project)

    point = ee.Geometry.Point([lon, lat])
    image = ee.Image(SRTM_ELEVATION_IMAGE).select(SRTM_ELEVATION_BAND)
    sample = image.sample(point, scale=30).first()

    value = sample.get(SRTM_ELEVATION_BAND)
    if value is None:
        raise EarthEngineAuthError(
            f"No elevation returned for ({lat:.4f}, {lon:.4f}). "
            "The point may fall outside the DEM coverage."
        )

    elevation = value.getInfo()
    if elevation is None:
        raise EarthEngineAuthError(
            f"Elevation sample was empty for ({lat:.4f}, {lon:.4f})."
        )

    return float(elevation)


def verify_ghana_connection(*, project: str | None = None) -> dict[str, Any]:
    """
    Verify Earth Engine by printing elevation at the Ghana test point.

    Returns a small result dict with coordinates and elevation in meters.
    """
    lon, lat = GHANA_TEST_POINT
    elevation_m = get_elevation_at_point(lon, lat, project=project)

    result = {
        "location": "Kumasi, Ghana (cocoa belt test point)",
        "longitude": lon,
        "latitude": lat,
        "elevation_m": elevation_m,
        "dem": SRTM_ELEVATION_IMAGE,
    }

    print(
        f"Earth Engine OK — elevation at {result['location']}\n"
        f"  Coordinates: {lat:.4f}°N, {abs(lon):.4f}°W\n"
        f"  Elevation:   {elevation_m:.1f} m ({SRTM_ELEVATION_IMAGE})"
    )
    return result


def main() -> int:
    """CLI entrypoint: authenticate (if needed) and run the Ghana elevation check."""
    try:
        verify_ghana_connection()
    except EarthEngineNotAuthenticatedError as exc:
        print(exc, file=sys.stderr)
        return 1
    except EarthEngineAuthError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ee.EEException as exc:
        print(f"Earth Engine API error: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
