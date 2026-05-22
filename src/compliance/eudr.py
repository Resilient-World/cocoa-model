"""
EU Deforestation Regulation (EU) 2023/1115 due diligence for cocoa operators.

Reference: EUR-Lex 32023R1115 (consolidated 26 December 2024). Applies from
30 December 2025 (Art. 32). Implements geolocation (Art. 2(28)), deforestation-free
attestation (Art. 3), due diligence (Arts. 8–10), and country risk (Art. 29).
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import ee
import numpy as np
import pycountry
import yaml
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

# ---------------------------------------------------------------------------
# Regulation constants
# ---------------------------------------------------------------------------

EUDR_APPLICABLE_FROM = date(2025, 12, 30)
POLYGON_REQUIRED_AREA_HA = 4.0
MIN_COORD_DECIMALS = 6
DEFAULT_BASELINE_DATE = "2020-12-31"
DEFAULT_COCOA_HS_CODE = "18010000"  # Cocoa beans, whole or broken, raw or roasted

HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"
JRC_GFC2020_ASSET = "JRC/GFC2020/V1"

# Art. 10 due-diligence criteria (a)–(n)
ART10_CRITERIA: tuple[str, ...] = tuple(chr(ord("a") + i) for i in range(14))

CountryRiskLevel = Literal["low", "standard", "high"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RISK_CONFIG = _REPO_ROOT / "config" / "eudr_country_risk.yaml"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PlotGeometry(BaseModel):
    """Production plot geolocation per Art. 2(28) and Annex II."""

    plot_id: str
    country: str = Field(..., min_length=3, max_length=3, description="ISO 3166-1 alpha-3")
    polygon: dict[str, Any] = Field(..., description="GeoJSON Geometry (Point or Polygon)")
    area_ha: float = Field(..., gt=0.0)
    producer_id: str
    production_start: date
    production_end: date

    @field_validator("country")
    @classmethod
    def _normalize_country(cls, value: str) -> str:
        code = value.strip().upper()
        if pycountry.countries.get(alpha_3=code) is None:
            raise ValueError(f"Unknown ISO 3166-1 alpha-3 country code: {code!r}")
        return code


class ValidationResult(BaseModel):
    """Outcome of :func:`validate_geolocation`."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    geometry_type: str | None = None
    min_decimal_places_observed: int | None = None


class DeforestationResult(BaseModel):
    """Forest-change screening for Art. 3 (deforestation-free)."""

    is_deforestation_free: bool
    loss_pixels: int
    loss_area_ha: float
    evidence_geotiff_path: str | None = None
    hansen_loss_detected: bool = False
    jrc_disturbance_detected: bool = False
    baseline_date: str = DEFAULT_BASELINE_DATE
    notes: list[str] = Field(default_factory=list)


class OperatorInfo(BaseModel):
    """Economic operator placing the DDS (Art. 4)."""

    operator_id: str
    name: str
    country: str = Field(..., min_length=3, max_length=3)
    role: Literal["operator", "trader"] = "operator"
    contact_email: str | None = None


class ProductInfo(BaseModel):
    """Relevant product description for Annex II."""

    description: str = "Cocoa beans"
    hs_code: str = DEFAULT_COCOA_HS_CODE
    net_mass_kg: float = Field(..., gt=0.0)
    species: str = "Theobroma cacao"


class RiskScore(BaseModel):
    """Art. 10 risk assessment summary (criteria a–n)."""

    criteria_scores: dict[str, float] = Field(
        ...,
        description="Normalised scores 0–1 per Art. 10 criterion (a–n)",
    )
    overall_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: CountryRiskLevel
    supply_chain_complexity: float = Field(..., ge=0.0, le=1.0)
    country_risk: CountryRiskLevel
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DueDiligenceStatement(BaseModel):
    """
    Due diligence statement (DDS) per Arts. 8–9 and Annex II.

    Supports JSON serialisation and the EU Information System CSV layout.
    """

    reference_number: str
    statement_date: date
    operator: OperatorInfo
    buyer_name: str | None = None
    supplier_name: str | None = None
    product: ProductInfo
    plot: PlotGeometry
    country_of_production: str
    geolocation_geojson: dict[str, Any]
    deforestation_free: bool
    deforestation_evidence_path: str | None = None
    country_risk: CountryRiskLevel
    risk_score: RiskScore
    geolocation_valid: bool
    validation_errors: list[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_eu_csv(self) -> str:
        """EU Information System–style flat CSV (one DDS row)."""
        row = {
            "ReferenceNumber": self.reference_number,
            "StatementDate": self.statement_date.isoformat(),
            "ActivityType": "PLACE_ON_MARKET",
            "OperatorName": self.operator.name,
            "OperatorCountry": self.operator.country,
            "OperatorRole": self.operator.role,
            "SupplierName": self.supplier_name or "",
            "BuyerName": self.buyer_name or "",
            "CountryOfProduction": self.country_of_production,
            "PlotId": self.plot.plot_id,
            "ProducerId": self.plot.producer_id,
            "ProductionStart": self.plot.production_start.isoformat(),
            "ProductionEnd": self.plot.production_end.isoformat(),
            "CommodityCode": self.product.hs_code,
            "ProductDescription": self.product.description,
            "NetMassKg": f"{self.product.net_mass_kg:.3f}",
            "Species": self.product.species,
            "GeolocationType": self.geolocation_geojson.get("type", ""),
            "GeolocationGeoJSON": json.dumps(self.geolocation_geojson, separators=(",", ":")),
            "AreaHa": f"{self.plot.area_ha:.4f}",
            "DeforestationFree": "TRUE" if self.deforestation_free else "FALSE",
            "CountryRisk": self.country_risk,
            "OverallRiskScore": f"{self.risk_score.overall_score:.4f}",
            "EvidenceGeotiff": self.deforestation_evidence_path or "",
        }
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# Geolocation validation (Art. 2(28))
# ---------------------------------------------------------------------------


def _shapely_geometry(plot: PlotGeometry) -> BaseGeometry:
    return shape(plot.polygon)


def _iter_coordinates(geom: BaseGeometry) -> list[tuple[float, float]]:
    if isinstance(geom, Point):
        return [(geom.x, geom.y)]
    if hasattr(geom, "geoms"):
        coords: list[tuple[float, float]] = []
        for part in geom.geoms:
            coords.extend(_iter_coordinates(part))
        return coords
    if hasattr(geom, "exterior"):
        return list(geom.exterior.coords)
    return list(geom.coords)


def _decimal_places(value: float) -> int:
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def _meets_min_precision(value: float, min_decimals: int = MIN_COORD_DECIMALS) -> bool:
    """True when the coordinate literal carries at least ``min_decimals`` fractional digits."""
    return _decimal_places(value) >= min_decimals


def validate_geolocation(plot: PlotGeometry) -> ValidationResult:
    """
    Validate plot geolocation for EUDR Art. 2(28).

    - All coordinates must use at least six decimal figures.
    - Plots larger than 4 ha require a polygon (not a single point).
    - Plots of 4 ha or less may use a single point.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        geom = _shapely_geometry(plot)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            errors=[f"Invalid GeoJSON geometry: {exc}"],
        )

    geom_type = geom.geom_type
    min_decimals = min(
        min(_decimal_places(x) for x, _ in _iter_coordinates(geom)),
        min(_decimal_places(y) for _, y in _iter_coordinates(geom)),
    )

    for lon, lat in _iter_coordinates(geom):
        if not _meets_min_precision(lon) or not _meets_min_precision(lat):
            errors.append(
                f"Coordinates must have at least {MIN_COORD_DECIMALS} decimal figures "
                f"(got lon={lon}, lat={lat})"
            )
            break

    if plot.area_ha > POLYGON_REQUIRED_AREA_HA:
        if geom_type not in ("Polygon", "MultiPolygon"):
            errors.append(
                f"Plots larger than {POLYGON_REQUIRED_AREA_HA} ha require polygon geolocation "
                f"(Art. 2(28)); got {geom_type}"
            )
    elif geom_type == "Point":
        warnings.append("Point geolocation accepted for plot area <= 4 ha (Art. 2(28))")

    if plot.production_end < plot.production_start:
        errors.append("production_end must be on or after production_start")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        geometry_type=geom_type,
        min_decimal_places_observed=min_decimals,
    )


# ---------------------------------------------------------------------------
# Deforestation screening (Art. 3) — Earth Engine backends
# ---------------------------------------------------------------------------


@dataclass
class _ForestScreening:
    loss_pixels: int
    loss_area_ha: float
    hansen_loss: bool
    jrc_disturbance: bool
    evidence_path: str | None


class ForestScreeningBackend(Protocol):
    """Pluggable backend for unit tests and offline runs."""

    def screen(
        self,
        geometry: dict[str, Any],
        *,
        baseline_date: str,
        plot_id: str,
    ) -> _ForestScreening: ...


class EarthEngineForestBackend:
    """Hansen GFC 2023 + JRC GFC2020 via Google Earth Engine."""

    def __init__(self) -> None:
        from data.gee_auth import initialize_earth_engine

        initialize_earth_engine()

    def screen(
        self,
        geometry: dict[str, Any],
        *,
        baseline_date: str,
        plot_id: str,
    ) -> _ForestScreening:
        ee_geom = ee.Geometry(geometry)
        baseline = date.fromisoformat(baseline_date)
        # Hansen lossyear: 1 = 2001 … 23 = 2023; losses after 2020 → year index >= 21
        min_loss_year_index = baseline.year - 2000 + 1

        hansen = ee.Image(HANSEN_ASSET)
        lossyear = hansen.select("lossyear")
        treecover = hansen.select("treecover2000")
        post_baseline_loss = lossyear.gte(min_loss_year_index).And(treecover.gt(0))

        jrc = ee.Image(JRC_GFC2020_ASSET)
        # Map1: forest disturbance / non-forest change layer in GFC2020 V1
        jrc_disturbance = jrc.select("Map1").gte(1)

        combined_loss = post_baseline_loss.Or(jrc_disturbance).selfMask()

        stats = combined_loss.reduceRegion(
            reducer=ee.Reducer.count().combine(
                reducer2=ee.Reducer.sum(),
                sharedInputs=True,
            ),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9,
        )
        stats_info = stats.getInfo() or {}
        loss_pixels = int(
            stats_info.get("lossyear_count", 0) or stats_info.get("Map1_count", 0) or 0
        )

        pixel_area_m2 = 30 * 30
        loss_area_ha = loss_pixels * pixel_area_m2 / 10_000.0

        hansen_only = post_baseline_loss.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9,
        ).getInfo()
        hansen_pixels = int((hansen_only or {}).get("lossyear", 0) or 0)

        jrc_only = jrc_disturbance.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9,
        ).getInfo()
        jrc_pixels = int((jrc_only or {}).get("Map1", 0) or 0)

        evidence_dir = Path("reports/eudr_evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = str(evidence_dir / f"{plot_id}_forest_loss.tif")

        try:
            url = combined_loss.getDownloadURL(
                {
                    "scale": 30,
                    "region": ee_geom,
                    "format": "GEO_TIFF",
                }
            )
            # Store manifest URL when full export is not run in API context
            evidence_path = f"{evidence_path}#download={url[:120]}"
        except Exception:
            evidence_path = None

        return _ForestScreening(
            loss_pixels=max(loss_pixels, hansen_pixels, jrc_pixels),
            loss_area_ha=loss_area_ha,
            hansen_loss=hansen_pixels > 0,
            jrc_disturbance=jrc_pixels > 0,
            evidence_path=evidence_path,
        )


class MockForestBackend:
    """Synthetic Hansen/JRC responses for tests."""

    def __init__(self, *, loss_by_plot: dict[str, _ForestScreening] | None = None) -> None:
        self._loss_by_plot = loss_by_plot or {}

    def screen(
        self,
        geometry: dict[str, Any],
        *,
        baseline_date: str,
        plot_id: str,
    ) -> _ForestScreening:
        del geometry, baseline_date
        if plot_id in self._loss_by_plot:
            return self._loss_by_plot[plot_id]
        return _ForestScreening(
            loss_pixels=0,
            loss_area_ha=0.0,
            hansen_loss=False,
            jrc_disturbance=False,
            evidence_path=None,
        )


def check_deforestation_free(
    plot: PlotGeometry,
    baseline_date: str = DEFAULT_BASELINE_DATE,
    *,
    backend: ForestScreeningBackend | None = None,
) -> DeforestationResult:
    """
      Screen plot for forest loss after ``baseline_date`` (default 31 Dec 2020).

      Uses Hansen Global Forest Change 2023 v1.11 and JRC Tropical Moist Forest
    GFC2020 V1 disturbance layer via Earth Engine unless ``backend`` is supplied.
    """
    screen_backend = backend or EarthEngineForestBackend()
    screening = screen_backend.screen(
        plot.polygon,
        baseline_date=baseline_date,
        plot_id=plot.plot_id,
    )
    is_free = screening.loss_pixels == 0
    notes: list[str] = []
    if screening.hansen_loss:
        notes.append("Hansen tree-cover loss detected after baseline")
    if screening.jrc_disturbance:
        notes.append("JRC GFC2020 disturbance class detected")

    return DeforestationResult(
        is_deforestation_free=is_free,
        loss_pixels=screening.loss_pixels,
        loss_area_ha=screening.loss_area_ha,
        evidence_geotiff_path=screening.evidence_path,
        hansen_loss_detected=screening.hansen_loss,
        jrc_disturbance_detected=screening.jrc_disturbance,
        baseline_date=baseline_date,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Country risk (Art. 29)
# ---------------------------------------------------------------------------


def _load_country_risk_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or _DEFAULT_RISK_CONFIG
    if not config_path.is_file():
        return {"default_risk": "standard", "countries": {}}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def assess_country_risk(
    country_iso3: str,
    *,
    config_path: Path | None = None,
) -> CountryRiskLevel:
    """
    Return country benchmark risk level (Art. 29).

    Loads ``config/eudr_country_risk.yaml``; unlisted countries receive
    ``default_risk`` (``standard`` per Art. 29(2)).
    """
    code = country_iso3.strip().upper()
    if pycountry.countries.get(alpha_3=code) is None:
        raise ValueError(f"Unknown ISO 3166-1 alpha-3 country code: {code!r}")

    config = _load_country_risk_config(config_path)
    default = config.get("default_risk", "standard")
    countries = config.get("countries") or {}
    entry = countries.get(code)
    if entry is None:
        return default  # type: ignore[return-value]
    risk = entry.get("risk", default) if isinstance(entry, dict) else entry
    if risk not in ("low", "standard", "high"):
        raise ValueError(f"Invalid risk level {risk!r} for country {code}")
    return risk  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Risk assessment (Art. 10)
# ---------------------------------------------------------------------------


_COUNTRY_RISK_WEIGHT = {"low": 0.15, "standard": 0.45, "high": 0.85}


def risk_assessment(
    plot: PlotGeometry,
    country_risk: CountryRiskLevel,
    supply_chain_complexity: float,
    *,
    deforestation_result: DeforestationResult | None = None,
) -> RiskScore:
    """
    Art. 10 due-diligence risk scoring across criteria (a)–(n).

    ``supply_chain_complexity`` is normalised 0 (simple) – 1 (highly complex).
    """
    complexity = float(np.clip(supply_chain_complexity, 0.0, 1.0))
    country_w = _COUNTRY_RISK_WEIGHT[country_risk]

    forest_risk = 0.0
    if deforestation_result is not None and not deforestation_result.is_deforestation_free:
        forest_risk = min(1.0, 0.5 + deforestation_result.loss_area_ha / max(plot.area_ha, 0.01))

    scores: dict[str, float] = {}
    for idx, key in enumerate(ART10_CRITERIA):
        if key == "a":
            scores[key] = forest_risk
        elif key in ("h", "i", "j"):
            scores[key] = complexity
        elif key == "k":
            scores[key] = country_w
        elif key == "g":
            scores[key] = 0.2 if plot.area_ha > POLYGON_REQUIRED_AREA_HA else 0.1
        else:
            base = 0.25 * country_w + 0.35 * complexity + 0.2 * forest_risk
            scores[key] = min(1.0, base + 0.02 * idx)

    overall = float(sum(scores.values()) / len(scores))
    if overall < 0.35:
        level: CountryRiskLevel = "low"
    elif overall < 0.65:
        level = "standard"
    else:
        level = "high"

    return RiskScore(
        criteria_scores=scores,
        overall_score=overall,
        risk_level=level,
        supply_chain_complexity=complexity,
        country_risk=country_risk,
    )


# ---------------------------------------------------------------------------
# DDS generation (Arts. 8–9, Annex II)
# ---------------------------------------------------------------------------


def generate_dds(
    plot: PlotGeometry,
    operator: OperatorInfo,
    product: ProductInfo,
    *,
    buyer_name: str | None = None,
    supplier_name: str | None = None,
    deforestation_result: DeforestationResult | None = None,
    country_risk: CountryRiskLevel | None = None,
    supply_chain_complexity: float = 0.3,
    reference_number: str | None = None,
) -> DueDiligenceStatement:
    """Build a due diligence statement with Annex II fields and risk annex."""
    validation = validate_geolocation(plot)
    if deforestation_result is None:
        deforestation_result = check_deforestation_free(plot)

    risk_level = country_risk or assess_country_risk(plot.country)
    risk = risk_assessment(
        plot,
        risk_level,
        supply_chain_complexity,
        deforestation_result=deforestation_result,
    )

    ref = reference_number or f"DDS-{uuid.uuid4().hex[:12].upper()}"

    return DueDiligenceStatement(
        reference_number=ref,
        statement_date=date.today(),
        operator=operator,
        buyer_name=buyer_name,
        supplier_name=supplier_name,
        product=product,
        plot=plot,
        country_of_production=plot.country,
        geolocation_geojson=plot.polygon,
        deforestation_free=deforestation_result.is_deforestation_free,
        deforestation_evidence_path=deforestation_result.evidence_geotiff_path,
        country_risk=risk_level,
        risk_score=risk,
        geolocation_valid=validation.is_valid,
        validation_errors=validation.errors,
    )
