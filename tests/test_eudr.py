"""Tests for :mod:`compliance.eudr` (EU Regulation 2023/1115)."""

from __future__ import annotations

import csv
import io
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from api.main import app
from compliance.eudr import (
    MockForestBackend,
    OperatorInfo,
    PlotGeometry,
    ProductInfo,
    _ForestScreening,
    assess_country_risk,
    check_deforestation_free,
    generate_dds,
    validate_geolocation,
)

# Western Côte d'Ivoire — illustrative deforestation hotspot (6-decimal polygon)
CDI_DEFORESTED_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-7.542123, 7.121456],
            [-7.541123, 7.121456],
            [-7.541123, 7.120456],
            [-7.542123, 7.120456],
            [-7.542123, 7.121456],
        ]
    ],
}

CDI_CLEAN_POINT = {
    "type": "Point",
    "coordinates": [-5.345678, 6.123456],
}

CDI_CLEAN_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-5.345678, 6.123456],
            [-5.344678, 6.123456],
            [-5.344678, 6.122456],
            [-5.345678, 6.122456],
            [-5.345678, 6.123456],
        ]
    ],
}


def _plot(
    *,
    polygon: dict,
    area_ha: float,
    plot_id: str = "TEST-001",
) -> PlotGeometry:
    return PlotGeometry(
        plot_id=plot_id,
        country="CIV",
        polygon=polygon,
        area_ha=area_ha,
        producer_id="PROD-001",
        production_start=date(2024, 10, 1),
        production_end=date(2025, 3, 31),
    )


def test_polygon_required_above_4ha() -> None:
    plot = _plot(polygon=CDI_CLEAN_POINT, area_ha=5.0)
    result = validate_geolocation(plot)
    assert not result.is_valid
    assert any("polygon" in err.lower() for err in result.errors)


def test_point_valid_at_or_below_4ha() -> None:
    plot = _plot(polygon=CDI_CLEAN_POINT, area_ha=3.5)
    result = validate_geolocation(plot)
    assert result.is_valid
    assert result.geometry_type == "Point"


def test_six_decimal_rule_rejects_low_precision() -> None:
    low_precision = {
        "type": "Point",
        "coordinates": [-5.12, 6.65],
    }
    plot = _plot(polygon=low_precision, area_ha=2.0)
    result = validate_geolocation(plot)
    assert not result.is_valid
    assert any("decimal" in err.lower() for err in result.errors)


def test_hansen_detection_on_cdi_deforested_plot_mock() -> None:
    plot = _plot(
        polygon=CDI_DEFORESTED_POLYGON,
        area_ha=2.8,
        plot_id="CIV-DEF-001",
    )
    backend = MockForestBackend(
        loss_by_plot={
            "CIV-DEF-001": _ForestScreening(
                loss_pixels=87,
                loss_area_ha=0.78,
                hansen_loss=True,
                jrc_disturbance=False,
                evidence_path="reports/eudr_evidence/CIV-DEF-001_forest_loss.tif",
            )
        }
    )
    result = check_deforestation_free(plot, backend=backend)
    assert not result.is_deforestation_free
    assert result.loss_pixels == 87
    assert result.hansen_loss_detected
    assert result.evidence_geotiff_path is not None


def test_dds_round_trip_json_and_csv() -> None:
    plot = _plot(polygon=CDI_CLEAN_POLYGON, area_ha=3.2, plot_id="CIV-CLEAN-001")
    operator = OperatorInfo(
        operator_id="OP-1",
        name="Cocoa Co-op Abidjan",
        country="CIV",
    )
    product = ProductInfo(net_mass_kg=12_500.0, hs_code="18010000")
    backend = MockForestBackend()

    dds = generate_dds(
        plot,
        operator,
        product,
        deforestation_result=check_deforestation_free(plot, backend=backend),
        country_risk="standard",
        supply_chain_complexity=0.25,
    )

    parsed = json.loads(dds.to_json())
    assert parsed["reference_number"] == dds.reference_number
    assert parsed["deforestation_free"] is True
    assert parsed["product"]["hs_code"] == "18010000"
    assert parsed["plot"]["country"] == "CIV"

    csv_text = dds.to_eu_csv()
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["ReferenceNumber"] == dds.reference_number
    assert rows[0]["CountryOfProduction"] == "CIV"
    assert rows[0]["DeforestationFree"] == "TRUE"
    geo = json.loads(rows[0]["GeolocationGeoJSON"])
    assert geo["type"] == "Polygon"


def test_assess_country_risk_defaults_and_config() -> None:
    assert assess_country_risk("NLD") == "standard"
    assert assess_country_risk("BRA") == "high"
    assert assess_country_risk("PER") == "low"


def test_risk_assessment_has_art10_criteria() -> None:
    plot = _plot(polygon=CDI_CLEAN_POLYGON, area_ha=3.0)
    from compliance.eudr import risk_assessment

    score = risk_assessment(plot, "standard", supply_chain_complexity=0.5)
    assert len(score.criteria_scores) == 14
    assert set(score.criteria_scores) == set("abcdefghijklmn")
    assert 0.0 <= score.overall_score <= 1.0


@pytest.fixture
def api_client() -> TestClient:
    with TestClient(app) as client:
        yield client


def test_compliance_dds_api_endpoint(api_client: TestClient) -> None:
    payload = {
        "plot": {
            "plot_id": "API-CIV-001",
            "country": "CIV",
            "polygon": CDI_CLEAN_POLYGON,
            "area_ha": 3.5,
            "producer_id": "PROD-API",
            "production_start": "2024-10-01",
            "production_end": "2025-03-31",
        },
        "operator": {
            "operator_id": "OP-API",
            "name": "Test Operator",
            "country": "CIV",
            "role": "operator",
        },
        "product": {
            "description": "Cocoa beans",
            "hs_code": "18010000",
            "net_mass_kg": 5000.0,
        },
        "supply_chain_complexity": 0.2,
        "use_gee_deforestation_check": False,
    }
    response = api_client.post("/compliance/dds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["geolocation_valid"] is True
    assert "dds_json" in data
    assert "dds_csv" in data
    assert data["risk_score"]["country_risk"] == "standard"
    dds_inner = json.loads(data["dds_json"])
    assert dds_inner["plot"]["plot_id"] == "API-CIV-001"


def test_compliance_dds_api_rejects_invalid_geolocation(api_client: TestClient) -> None:
    payload = {
        "plot": {
            "plot_id": "API-BAD",
            "country": "CIV",
            "polygon": {"type": "Point", "coordinates": [-5.1, 6.6]},
            "area_ha": 10.0,
            "producer_id": "P1",
            "production_start": "2024-10-01",
            "production_end": "2025-03-31",
        },
        "operator": {
            "operator_id": "OP",
            "name": "Op",
            "country": "CIV",
        },
        "product": {"net_mass_kg": 100.0},
    }
    response = api_client.post("/compliance/dds", json=payload)
    assert response.status_code == 400
