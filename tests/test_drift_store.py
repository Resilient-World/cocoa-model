"""Tests for drift monitoring persistence."""

from __future__ import annotations

from pathlib import Path

from monitoring.drift_store import DriftStore, DriftStratumState
from monitoring.wctm import WeightedConformalTestMartingale


def test_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "drift.json"
    store = DriftStore(state_path=path, alpha_fpr=0.01)
    key = "ssp245:2050:ghana"
    w = store.get_wctm(key)
    for _ in range(30):
        w.update(4.0)
    x = store.get_x_wctm(key)
    for _ in range(10):
        x.update(2.0)
    c = store.get_cusum(key)
    c.update(1.0)
    store.save_after_update(
        key,
        wctm=w,
        x_wctm=x,
        cusum=c,
        diagnosis="none",
        alarm_at=None,
        feature_ema=[0.1] * 8,
    )

    store2 = DriftStore(state_path=path, alpha_fpr=0.01)
    st = store2.get_stratum_state(key)
    assert st.update_count == 1
    assert st.log_martingale > 0
    w2 = store2.get_wctm(key)
    assert len(w2._calibration_scores) >= 1


def test_stratum_isolation(tmp_path: Path) -> None:
    path = tmp_path / "drift.json"
    store = DriftStore(state_path=path)
    w1 = store.get_wctm("ssp245:2030:ghana")
    w2 = store.get_wctm("ssp585:2080:civ")
    for _ in range(40):
        w1.update(4.0)
    for _ in range(5):
        w2.update(0.1)
    store.save_after_update(
        "ssp245:2030:ghana",
        wctm=w1,
        x_wctm=WeightedConformalTestMartingale(),
        cusum=store.get_cusum("ssp245:2030:ghana"),
        diagnosis="none",
        alarm_at=None,
    )
    assert (
        store.get_wctm("ssp585:2080:civ").log_martingale
        < store.get_wctm("ssp245:2030:ghana").log_martingale
    )


def test_drift_stratum_state_serialization() -> None:
    st = DriftStratumState(log_martingale=1.2, last_diagnosis="concept_shift")
    d = st.to_dict()
    st2 = DriftStratumState.from_dict(d)
    assert st2.log_martingale == 1.2
    assert st2.last_diagnosis == "concept_shift"
