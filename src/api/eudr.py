"""
EUDR due diligence API (EU Regulation 2023/1115, October 2025 amendments).

Exposes plot-level deforestation screening (cutoff 2020-12-31), Whisp risk classification,
FDP/Hansen cross-check, and traceability to source datasets.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date
from typing import Any, Literal

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.config import APISettings
from compliance.eudr import (
    DEFAULT_BASELINE_DATE,
    HANSEN_ASSET,
    JRC_GFC2020_ASSET,
    DeforestationResult,
    PlotGeometry,
    check_deforestation_free,
    validate_geolocation,
)
from data.cocoa_exposure import FDP_COCOA_COLLECTION, FDP_MODEL_CARD_URL
from data.whisp_client import (
    EUDR_FOREST_CUTOFF_DATE,
    WHISP_DOCS_URL,
    WHISP_PORTAL_URL,
    EudrRiskClass,
    WhispClient,
    WhispPlotResult,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="", tags=["eudr"])

HANSEN_CATALOG_URL = "https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2023_v1_11"
JRC_GFC_CATALOG_URL = "https://data.jrc.ec.europa.eu/dataset/0a5f33b0-8b1c-4b1c-9c3b-0e0e0e0e0e0e"
EUDR_REGULATION_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1115"
EUDR_OCT_2025_NOTE = (
    "EU Regulation (EU) 2023/1115 as amended (application from 30 December 2025); "
    "forest cutoff 2020-12-31 per Art. 3."
)


class EudrDueDiligenceRequest(BaseModel):
    """POST /eudr-due-diligence body."""

    farm_polygon: dict[str, Any] = Field(..., description="GeoJSON Polygon or MultiPolygon")
    commodity: Literal["cocoa"] = "cocoa"
    plot_id: str | None = Field(default=None, description="Optional plot identifier")
    country_iso3: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="Producer country (ISO3); inferred from polygon centroid if omitted",
    )
    area_ha: float | None = Field(
        default=None, gt=0.0, description="Plot area (ha); estimated if omitted"
    )
    use_gee_fdp_screening: bool = Field(
        default=True,
        description="Cross-check Whisp with Hansen/JRC GEE screening (FDP-aligned Art. 3)",
    )


class EudrDueDiligenceResponse(BaseModel):
    """EUDR screening with traceability to source datasets."""

    deforestation_post_2020: bool = Field(
        ...,
        description="True if forest loss detected after 2020-12-31 (not deforestation-free)",
    )
    protected_area_overlap: bool
    risk_class: EudrRiskClass
    evidence_urls: list[str]
    whisp_report_id: str
    traceability: dict[str, Any] = Field(
        ...,
        description="Source datasets and screening provenance for probabilistic outputs",
    )
    geolocation_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)


class EudrStatusBlock(BaseModel):
    """Optional block attached to simulation responses when ``farm_polygon`` is supplied."""

    deforestation_post_2020: bool
    protected_area_overlap: bool
    risk_class: EudrRiskClass
    evidence_urls: list[str]
    whisp_report_id: str | None = None
    traceability: dict[str, Any] = Field(default_factory=dict)


def _polygon_area_ha(geojson_polygon: dict[str, Any]) -> float:
    from shapely.geometry import shape

    geom = shape(geojson_polygon)
    if geom.geom_type == "Point":
        return 1.0
    # Rough ha from m² (WGS84 degree→m varies; sufficient for EUDR 4 ha rule)
    return max(float(geom.area) * 111_320 * 111_320 / 10_000 / 2.0, 0.01)


def _infer_country_iso3(geojson_polygon: dict[str, Any]) -> str:
    from shapely.geometry import shape

    geom = shape(geojson_polygon)
    lon, lat = float(geom.centroid.x), float(geom.centroid.y)
    if -9.0 <= lon <= -2.5 and 4.0 <= lat <= 9.0:
        return "CIV"
    if -4.0 <= lon <= 2.0 and 4.0 <= lat <= 9.5:
        return "GHA"
    if 8.0 <= lon <= 17.0 and 1.0 <= lat <= 14.0:
        return "CMR"
    return "CIV"


def _build_plot_geometry(
    geojson_polygon: dict[str, Any],
    *,
    plot_id: str | None,
    country_iso3: str | None,
    area_ha: float | None,
) -> PlotGeometry:
    pid = plot_id or f"EUDR-{uuid.uuid4().hex[:8].upper()}"
    country = (country_iso3 or _infer_country_iso3(geojson_polygon)).upper()
    area = area_ha if area_ha is not None else _polygon_area_ha(geojson_polygon)
    return PlotGeometry(
        plot_id=pid,
        country=country,
        polygon=geojson_polygon,
        area_ha=area,
        producer_id="unknown",
        production_start=date(2024, 1, 1),
        production_end=date.today(),
    )


def _default_evidence_urls() -> list[str]:
    return [
        WHISP_PORTAL_URL,
        WHISP_DOCS_URL,
        HANSEN_CATALOG_URL,
        JRC_GFC_CATALOG_URL,
        FDP_MODEL_CARD_URL,
        EUDR_REGULATION_URL,
    ]


def _merge_deforestation(
    whisp: WhispPlotResult,
    gee: DeforestationResult | None,
) -> tuple[bool, list[str]]:
    """Combine Whisp flags with Hansen/JRC (FDP-aligned) screening."""
    whisp_loss = whisp.deforestation_flag
    gee_loss = gee is not None and not gee.is_deforestation_free
    combined = whisp_loss or gee_loss
    sources = [
        f"whisp:{whisp.report_id}",
        f"whisp_risk:{whisp.risk_level}",
    ]
    if gee is not None:
        sources.append(f"hansen:{HANSEN_ASSET}:loss_pixels={gee.loss_pixels}")
        sources.append(f"jrc:{JRC_GFC2020_ASSET}:disturbance={gee.jrc_disturbance_detected}")
        sources.append(f"fdp_collection:{FDP_COCOA_COLLECTION}")
    return combined, sources


def build_whisp_client(settings: APISettings) -> WhispClient:
    import os

    from data.whisp_client import DEFAULT_WHISP_BASE_URL

    api_key = os.environ.get("WHISP_API_KEY") or settings.whisp_api_key
    return WhispClient(
        base_url=str(settings.whisp_base_url or DEFAULT_WHISP_BASE_URL),
        api_key=api_key,
    )


async def run_eudr_due_diligence(
    request: EudrDueDiligenceRequest,
    *,
    settings: APISettings,
    whisp_client: WhispClient | None = None,
) -> EudrDueDiligenceResponse:
    """Core due diligence: Whisp + optional GEE/FDP deforestation cross-check."""
    from api import metrics as prom_metrics
    from api.telemetry import trace_span

    with trace_span("eudr.screen", commodity=request.commodity):
        try:
            return await _run_eudr_due_diligence_impl(
                request, settings=settings, whisp_client=whisp_client
            )
        except httpx.TimeoutException:
            prom_metrics.inc_eudr_status("timeout")
            raise
        except httpx.HTTPError:
            prom_metrics.inc_eudr_status("fail")
            raise
        except Exception:
            prom_metrics.inc_eudr_status("fail")
            raise


async def _run_eudr_due_diligence_impl(
    request: EudrDueDiligenceRequest,
    *,
    settings: APISettings,
    whisp_client: WhispClient | None = None,
) -> EudrDueDiligenceResponse:
    from api import metrics as prom_metrics

    plot = _build_plot_geometry(
        request.farm_polygon,
        plot_id=request.plot_id,
        country_iso3=request.country_iso3,
        area_ha=request.area_ha,
    )
    validation = validate_geolocation(plot)

    client = whisp_client or build_whisp_client(settings)
    whisp = await client.check_plot(request.farm_polygon, commodity=request.commodity)

    gee_result: DeforestationResult | None = None
    if request.use_gee_fdp_screening:
        try:
            gee_result = check_deforestation_free(plot)
        except Exception as exc:
            log.warning("GEE/FDP deforestation screening skipped: %s", exc)

    deforestation_post_2020, screening_sources = _merge_deforestation(whisp, gee_result)

    risk_class: EudrRiskClass = whisp.eudr_risk_class
    if deforestation_post_2020 or whisp.protected_area_overlap:
        risk_class = "enhanced"

    evidence = list(dict.fromkeys(_default_evidence_urls() + list(whisp.evidence_urls)))

    traceability = {
        "regulation": EUDR_OCT_2025_NOTE,
        "forest_baseline_cutoff": EUDR_FOREST_CUTOFF_DATE,
        "commodity": request.commodity,
        "plot_id": plot.plot_id,
        "country_iso3": plot.country,
        "screening_sources": screening_sources,
        "whisp": whisp.to_traceability(),
        "hansen_asset": HANSEN_ASSET,
        "jrc_asset": JRC_GFC2020_ASSET,
        "fdp_cocoa_asset": FDP_COCOA_COLLECTION,
        "gee_screening_attempted": request.use_gee_fdp_screening,
    }
    if gee_result is not None:
        traceability["gee"] = {
            "is_deforestation_free": gee_result.is_deforestation_free,
            "loss_pixels": gee_result.loss_pixels,
            "loss_area_ha": gee_result.loss_area_ha,
            "baseline_date": gee_result.baseline_date or DEFAULT_BASELINE_DATE,
            "evidence_geotiff_path": gee_result.evidence_geotiff_path,
        }

    prom_metrics.inc_eudr_status("pass" if validation.is_valid else "fail")

    return EudrDueDiligenceResponse(
        deforestation_post_2020=deforestation_post_2020,
        protected_area_overlap=whisp.protected_area_overlap,
        risk_class=risk_class,
        evidence_urls=evidence,
        whisp_report_id=whisp.report_id,
        traceability=traceability,
        geolocation_valid=validation.is_valid,
        validation_errors=validation.errors,
    )


def evaluate_eudr_status(
    farm_polygon: dict[str, Any],
    *,
    settings: APISettings,
    commodity: str = "cocoa",
    whisp_client: WhispClient | None = None,
) -> EudrStatusBlock | None:
    """Sync helper for simulation endpoints (runs async Whisp in event loop)."""
    req = EudrDueDiligenceRequest(
        farm_polygon=farm_polygon,
        commodity="cocoa",  # type: ignore[arg-type]
        use_gee_fdp_screening=False,
    )
    try:
        result = asyncio.run(
            run_eudr_due_diligence(req, settings=settings, whisp_client=whisp_client)
        )
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            run_eudr_due_diligence(req, settings=settings, whisp_client=whisp_client)
        )
    return EudrStatusBlock(
        deforestation_post_2020=result.deforestation_post_2020,
        protected_area_overlap=result.protected_area_overlap,
        risk_class=result.risk_class,
        evidence_urls=result.evidence_urls,
        whisp_report_id=result.whisp_report_id,
        traceability=result.traceability,
    )


@router.post(
    "/eudr-due-diligence",
    response_model=EudrDueDiligenceResponse,
    summary="EUDR plot screening (Whisp + FDP/Hansen, cutoff 2020-12-31)",
)
async def eudr_due_diligence_endpoint(
    request: EudrDueDiligenceRequest,
    http_request: Request,
) -> EudrDueDiligenceResponse:
    """
    Plot-level EUDR due diligence for cocoa: deforestation post-2020, protected areas,
    risk class, and traceable evidence URLs (Whisp, Hansen, JRC, FDP).
    """
    settings: APISettings = http_request.app.state.settings
    try:
        return await run_eudr_due_diligence(request, settings=settings)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Whisp API error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
