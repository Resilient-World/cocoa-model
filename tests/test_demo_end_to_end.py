"""End-to-end demo script with mocked GEE / Whisp (no network)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from data.whisp_client import MockWhispClient, WhispPlotResult
from scripts import demo_end_to_end as demo


@pytest.fixture
def mock_whisp() -> MockWhispClient:
    return MockWhispClient(
        default_result=WhispPlotResult(
            report_id="mock-e2e-whisp",
            deforestation_flag=False,
            protected_area_overlap=False,
            risk_level="low",
            eudr_risk_class="standard",
            evidence_urls=("https://whisp.openforis.org/report/mock-e2e-whisp",),
            source_datasets=("WHISP", "Hansen GFC", "JRC GFC2020"),
        )
    )


def test_run_end_to_end_demo_mock_gee(tmp_path, mock_whisp: MockWhispClient) -> None:
    import asyncio
    from api.config import APISettings

    settings = APISettings(
        era5_zarr_path=tmp_path / "era5.zarr",
        cmip6_zarr_path=tmp_path / "cmip6.zarr",
        era5_counterfactual_zarr_path=tmp_path / "era5_cf.zarr",
        use_real_features=False,
    )

    with patch(
        "scripts.demo_end_to_end.sample_cocoa_probability_at_point",
        return_value=0.72,
    ):
        payload = asyncio.run(
            demo.run_end_to_end_demo(
                settings=settings,
                whisp_client=mock_whisp,
                mock_gee=True,
                farm_size_ha=2.5,
                current_yield_t_ha=1.5,
            )
        )

    assert payload["climate_attributed_loss_t_per_ha"] >= 0.0
    assert payload["intervention_avoided_loss_t_per_ha"] >= 0.0
    assert payload["total_avoided_loss_usd"]["level"] == 0.9
    assert payload["total_avoided_loss_usd"]["ci_low"] <= payload["total_avoided_loss_usd"]["point"]
    assert payload["total_avoided_loss_usd"]["ci_high"] >= payload["total_avoided_loss_usd"]["point"]
    assert payload["cocoa_exposure_probability"] == pytest.approx(0.72)
    assert payload["eudr_status"]["risk_class"] == "standard"
    assert payload["eudr_status"]["deforestation_post_2020"] is False

    ids = {a["id"] for a in payload["source_attributions"]}
    assert {"whisp", "fdp_cocoa_2025a", "era5_land", "cmip6", "attrici", "casej_surrogate"} <= ids

    assert "scenario_ssp585_2050" in payload
    assert payload["scenario_ssp585_2050"]["avoided_loss_tonnes"]["mean"] >= 0.0


def test_demo_main_writes_json(tmp_path) -> None:
    out = tmp_path / "demo.json"
    stub = {
        "climate_attributed_loss_t_per_ha": 0.1,
        "intervention_avoided_loss_t_per_ha": 0.2,
        "total_avoided_loss_usd": {"point": 100.0, "ci_low": 50.0, "ci_high": 150.0, "level": 0.9, "method": "mcd"},
        "eudr_status": {},
        "source_attributions": [],
    }
    with patch("scripts.demo_end_to_end.asyncio.run", return_value=stub):
        rc = demo.main(["--mock-gee", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["climate_attributed_loss_t_per_ha"] == 0.1


def test_write_era5_and_attrici_zarr_roundtrip(tmp_path) -> None:
    era5 = tmp_path / "era5.zarr"
    cf = tmp_path / "cf.zarr"
    demo.write_era5_demo_zarr(era5, demo.SAMPLE_LAT, demo.SAMPLE_LON)
    demo.write_attrici_counterfactual_zarr(era5, cf, lat=demo.SAMPLE_LAT, lon=demo.SAMPLE_LON)
    assert era5.is_dir()
    assert cf.is_dir()
