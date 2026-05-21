"""Integration tests for WCTM on /simulate-scenario."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from api.config import APISettings
from api.schemas import DriftAlarmPayload, DriftStatus
from monitoring.drift_store import DriftStore


SCENARIO_PAYLOAD = {
    "farm_location": {"lat": 6.5, "lon": -1.2},
    "farm_size_ha": 5.0,
    "current_yield": 2.0,
    "intervention_type": "shade_trees",
    "cocoa_price_usd": 3200.0,
    "scenario": "ssp245",
    "horizon_year": 2050,
}


def test_drift_status_endpoint_unknown_stratum(tmp_path: Path) -> None:
    from api.drift_monitoring import get_drift_status_for_stratum
    store = DriftStore(state_path=tmp_path / "drift.json")
    with pytest.raises(ValueError, match="Invalid stratum"):
        get_drift_status_for_stratum("bad", drift_store=store)


def test_drift_status_payload_shape(tmp_path: Path) -> None:
    from api.drift_monitoring import get_drift_status_for_stratum

    store = DriftStore(state_path=tmp_path / "drift.json")
    key = "ssp245:2050:ghana"
    status = get_drift_status_for_stratum(key, drift_store=store)
    assert status.stratum_key == key
    assert status.diagnosis == "none"
    assert status.alarm_active is False


def test_apply_drift_monitoring_returns_status(tmp_path: Path) -> None:
    from api.drift_monitoring import apply_drift_monitoring
    from api.schemas import SimulateScenarioRequest

    request = SimulateScenarioRequest.model_validate(SCENARIO_PAYLOAD)
    settings = APISettings(drift_enabled=True, drift_state_path=tmp_path / "drift.json")
    drift_store = DriftStore(state_path=settings.drift_state_path)
    alarm, status, inflate = apply_drift_monitoring(
        request,
        y_obs=2.0,
        y_pred=1.5,
        interval_lo=1.0,
        interval_hi=2.0,
        drift_store=drift_store,
        conformal_store=None,
        settings=settings,
        climate_projected=torch.randn(1, 365, 11),
        static_factual=torch.randn(1, 13),
    )
    assert status is not None
    assert status.stratum_key == "ssp245:2050:ghana"
    assert isinstance(inflate, bool)


def test_concept_shift_inflates_interval_width(tmp_path: Path) -> None:
    from api.scenario_conformal import apply_scenario_conformal
    from api.schemas import SimulateScenarioRequest
    from tests.test_api_scenario_online import (
        N_CLIMATE_CHANNELS,
        SITE_STATIC_DIM,
        _mock_cqr_model,
    )

    request = SimulateScenarioRequest.model_validate(SCENARIO_PAYLOAD)
    climate = torch.randn(1, 365, N_CLIMATE_CHANNELS)
    static = torch.randn(1, SITE_STATIC_DIM)
    settings = APISettings(
        conformal_method="aci",
        drift_enabled=True,
        drift_inflation_factor=2.0,
    )
    store_path = tmp_path / "conf.json"
    drift_path = tmp_path / "drift.json"
    from api.online_conformal_store import OnlineConformalStore

    store = OnlineConformalStore(state_path=store_path, aci_eta=0.05)
    drift_store = DriftStore(state_path=drift_path, alpha_fpr=0.01)
    model = _mock_cqr_model()

    def _no_drift(*a, **k):
        return None, DriftStatus(
            stratum_key="ssp245:2050:ghana",
            log_martingale=0.0,
            alarm_active=False,
            diagnosis="none",
        ), False

    with patch("api.scenario_conformal.apply_drift_monitoring", _no_drift):
        base = apply_scenario_conformal(
            request,
            cqr_model=model,
            cqr_calibrator=None,
            store=store,
            drift_store=drift_store,
            settings=settings,
            climate_baseline=climate,
            climate_projected=climate,
            static_cf=static,
            static_factual=static,
            biotic_cf_frac=1.0,
            biotic_fact_frac=1.0,
        )

    def _inflate(*a, **k):
        return (
            DriftAlarmPayload(
                type="concept_shift",
                log_martingale=6.0,
                triggered_at="2026-05-20T00:00:00+00:00",
            ),
            DriftStatus(
                stratum_key="ssp245:2050:ghana",
                log_martingale=6.0,
                alarm_active=True,
                diagnosis="concept_shift",
            ),
            True,
        )

    store._updater_cache.clear()
    with patch("api.scenario_conformal.apply_drift_monitoring", _inflate):
        wide = apply_scenario_conformal(
            request,
            cqr_model=model,
            cqr_calibrator=None,
            store=store,
            drift_store=drift_store,
            settings=settings,
            climate_baseline=climate,
            climate_projected=climate,
            static_cf=static,
            static_factual=static,
            biotic_cf_frac=1.0,
            biotic_fact_frac=1.0,
        )

    assert base is not None and wide is not None
    assert (wide.ci_upper - wide.ci_lower) >= (base.ci_upper - base.ci_lower)