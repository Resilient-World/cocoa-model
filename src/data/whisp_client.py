"""
Async client for the Open Foris Whisp API (https://whisp.openforis.org).

Whisp assesses plot-level deforestation and land-use risk using Google Earth Engine
layers. This module never imports ``openforis_whisp`` directly — HTTP only — so the
MIT-licensed API can run without the Whisp Python stack.

Reference: Open Foris Whisp; EU Regulation (EU) 2023/1115 (EUDR), as amended October 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_WHISP_BASE_URL = "https://whisp.openforis.org"
EUDR_FOREST_CUTOFF_DATE = "2020-12-31"
WHISP_PORTAL_URL = "https://whisp.openforis.org/"
WHISP_DOCS_URL = "https://whisp.openforis.org/documentation/api-guide"

WhispRiskLevel = Literal["low", "medium", "high", "unknown"]
EudrRiskClass = Literal["standard", "enhanced"]


@dataclass(frozen=True)
class WhispPlotResult:
    """Normalized Whisp outcome for one plot geometry."""

    report_id: str
    deforestation_flag: bool
    protected_area_overlap: bool
    risk_level: WhispRiskLevel
    eudr_risk_class: EudrRiskClass
    forest_loss_ha: float | None = None
    commodity: str = "cocoa"
    baseline_date: str = EUDR_FOREST_CUTOFF_DATE
    evidence_urls: tuple[str, ...] = ()
    source_datasets: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    def to_traceability(self) -> dict[str, Any]:
        """Probabilistic / screening outputs with source attribution for DDS."""
        return {
            "whisp_report_id": self.report_id,
            "deforestation_flag": self.deforestation_flag,
            "protected_area_overlap": self.protected_area_overlap,
            "risk_level": self.risk_level,
            "eudr_risk_class": self.eudr_risk_class,
            "baseline_date": self.baseline_date,
            "evidence_urls": list(self.evidence_urls),
            "source_datasets": list(self.source_datasets),
            "whisp_portal": WHISP_PORTAL_URL,
        }


def _risk_level_to_eudr_class(level: WhispRiskLevel) -> EudrRiskClass:
    if level in ("high", "medium"):
        return "enhanced"
    return "standard"


def _parse_whisp_payload(payload: dict[str, Any], *, report_id: str) -> WhispPlotResult:
    """Map Whisp API JSON (flexible schema) to :class:`WhispPlotResult`."""
    risk_raw = str(
        payload.get("risk_class")
        or payload.get("risk_level")
        or payload.get("whisp_risk")
        or payload.get("overall_risk")
        or "unknown"
    ).lower()

    if risk_raw in ("high", "h", "3"):
        risk_level: WhispRiskLevel = "high"
    elif risk_raw in ("medium", "med", "m", "2"):
        risk_level = "medium"
    elif risk_raw in ("low", "l", "1"):
        risk_level = "low"
    else:
        risk_level = "unknown"

    deforestation_flag = bool(
        payload.get("deforestation")
        or payload.get("deforestation_flag")
        or payload.get("forest_loss")
        or payload.get("tree_loss")
        or (risk_level == "high")
    )
    protected = bool(
        payload.get("protected_area")
        or payload.get("protected_area_overlap")
        or payload.get("protected_overlap")
        or payload.get("in_protected_area")
    )

    forest_loss_ha = payload.get("forest_loss_ha") or payload.get("loss_area_ha")
    try:
        loss_ha = float(forest_loss_ha) if forest_loss_ha is not None else None
    except (TypeError, ValueError):
        loss_ha = None

    sources = payload.get("source_datasets") or payload.get("datasets") or []
    if isinstance(sources, str):
        sources = [sources]
    evidence = payload.get("evidence_urls") or payload.get("sources") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    return WhispPlotResult(
        report_id=report_id,
        deforestation_flag=deforestation_flag,
        protected_area_overlap=protected,
        risk_level=risk_level,
        eudr_risk_class=_risk_level_to_eudr_class(risk_level),
        forest_loss_ha=loss_ha,
        evidence_urls=tuple(str(u) for u in evidence),
        source_datasets=tuple(str(s) for s in sources),
        raw=payload,
    )


class WhispClient:
    """
    Async HTTP client for Whisp plot checks.

    Parameters
    ----------
    base_url:
        Whisp deployment root (default production).
    api_key:
        Bearer/API key from Whisp user profile (optional for public temp-key flows).
    timeout_s:
        HTTP timeout per request.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_WHISP_BASE_URL,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    async def check_plot(
        self,
        geojson_polygon: dict[str, Any],
        *,
        commodity: str = "cocoa",
        national_codes: list[str] | None = None,
    ) -> WhispPlotResult:
        """
        Submit a GeoJSON geometry and return deforestation / protected-area flags.

        The client posts to ``/api/submit/geojson`` and polls ``/api/report/{id}`` when
        the submission returns a pending token.
        """
        feature = {
            "type": "Feature",
            "geometry": geojson_polygon,
            "properties": {"commodity": commodity, "id": geojson_polygon.get("id", "plot-1")},
        }
        body: dict[str, Any] = {
            "geojson": {"type": "FeatureCollection", "features": [feature]},
            "commodity": commodity,
            "baseline_date": EUDR_FOREST_CUTOFF_DATE,
        }
        if national_codes:
            body["national_codes"] = national_codes

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            submit = await client.post(
                f"{self.base_url}/api/submit/geojson",
                json=body,
                headers=self._headers(),
            )
            submit.raise_for_status()
            submitted = submit.json()

            report_id = str(
                submitted.get("reportId")
                or submitted.get("report_id")
                or submitted.get("token")
                or submitted.get("id")
                or "whisp-pending"
            )

            if submitted.get("status") == "complete" or "results" in submitted:
                results = submitted.get("results") or submitted
                if isinstance(results, list) and results:
                    row = results[0] if isinstance(results[0], dict) else {"risk_level": "unknown"}
                elif isinstance(results, dict):
                    row = results
                else:
                    row = submitted
                return _parse_whisp_payload(row, report_id=report_id)

            report = await client.get(
                f"{self.base_url}/api/report/{report_id}",
                headers=self._headers(),
            )
            report.raise_for_status()
            data = report.json()
            rows = data.get("results") or data.get("plots") or data
            if isinstance(rows, list) and rows:
                row = rows[0] if isinstance(rows[0], dict) else {}
            elif isinstance(rows, dict):
                row = rows
            else:
                row = data
            return _parse_whisp_payload(row, report_id=report_id)

    async def aclose(self) -> None:
        """No persistent connection pool; kept for interface symmetry."""


class MockWhispClient(WhispClient):
    """Deterministic Whisp responses for tests (no HTTP)."""

    def __init__(
        self,
        *,
        results_by_plot: dict[str, WhispPlotResult] | None = None,
        default_result: WhispPlotResult | None = None,
    ) -> None:
        super().__init__(base_url="https://mock.whisp.test")
        self._results_by_plot = results_by_plot or {}
        self._default = default_result or WhispPlotResult(
            report_id="mock-whisp-001",
            deforestation_flag=False,
            protected_area_overlap=False,
            risk_level="low",
            eudr_risk_class="standard",
            evidence_urls=(WHISP_PORTAL_URL,),
            source_datasets=("WHISP", "Hansen GFC", "JRC GFC2020"),
        )

    async def check_plot(
        self,
        geojson_polygon: dict[str, Any],
        *,
        commodity: str = "cocoa",
        national_codes: list[str] | None = None,
    ) -> WhispPlotResult:
        del commodity, national_codes
        plot_id = str(geojson_polygon.get("id") or "default")
        return self._results_by_plot.get(plot_id, self._default)
