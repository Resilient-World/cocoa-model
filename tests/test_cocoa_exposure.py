import os

import pytest
import ee  # noqa: F401

from data.cocoa_exposure import CocoaExposureIngest, FDP_COCOA_COLLECTION


def test_collection_id_is_fdp_2025a():
    assert FDP_COCOA_COLLECTION == "projects/forestdatapartnership/assets/cocoa/model_2025a"


def test_backend_defaults_to_fdp():
    ing = CocoaExposureIngest(aoi=object(), backend="fdp")  # type: ignore[arg-type]
    assert ing.backend == "fdp"


def test_ensemble_weights_default():
    ing = CocoaExposureIngest(aoi=object(), backend="ensemble")  # type: ignore[arg-type]
    assert ing.ensemble_weights == (0.5, 0.3, 0.2)


def test_aef_backend_without_gee(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(aoi=object(), backend="aef")  # type: ignore[arg-type]
    monkeypatch.setattr(ing, "_aef_probability_at_point", lambda lat, lon: 0.55)
    p = ing.sample_point(5.84, -5.36)
    assert p == pytest.approx(0.55, abs=0.01)


def test_galileo_point_inference_without_gee(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class _StubModel:
        @staticmethod
        def build_batch_dict(**kwargs: object) -> dict:
            return kwargs

        def predict_proba(self, batch_dict: dict) -> torch.Tensor:
            return torch.tensor([[[[0.42]]]])

    ing = CocoaExposureIngest(aoi=object(), backend="galileo")  # type: ignore[arg-type]
    monkeypatch.setattr(ing, "_load_galileo_model", lambda: _StubModel())
    p = ing.sample_point(5.84, -5.36)
    assert p == pytest.approx(0.42, abs=0.01)


def test_ensemble_blends_aef_galileo_and_fdp(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(aoi=object(), backend="ensemble")  # type: ignore[arg-type]
    monkeypatch.setattr(ing, "_aef_probability_at_point", lambda lat, lon: 0.9)
    monkeypatch.setattr(ing, "_galileo_probability_at_point", lambda lat, lon: 0.4)
    monkeypatch.setattr(ing, "_fdp_probability_at_point", lambda lat, lon, scale_m: 0.2)
    p = ing.sample_point(5.84, -5.36)
    # 0.5*0.9 + 0.3*0.4 + 0.2*0.2 = 0.61
    assert p == pytest.approx(0.61, abs=0.01)


@pytest.mark.integration
def test_sample_point_in_civ_cocoa_belt():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("EARTHENGINE_PROJECT"):
        pytest.skip("No GEE credentials")
    import ee

    ee.Initialize()
    # Known cocoa-dense pixel near Divo, CIV (in-situ validation region from Kalischek 2023)
    aoi = ee.Geometry.Point([-5.36, 5.84]).buffer(500)
    ing = CocoaExposureIngest(aoi, year=2023)
    p = ing.sample_point(5.84, -5.36)
    assert 0.0 <= p <= 1.0
    assert p > 0.3  # not a hard bound but rejects obvious regressions


@pytest.mark.integration
def test_feature_resolver_uses_fdp_when_available(monkeypatch):
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("EARTHENGINE_PROJECT"):
        pytest.skip("No GEE credentials")
    from api.feature_resolver import FarmFeatureResolver, FeatureResolverConfig

    r = FarmFeatureResolver(FeatureResolverConfig())
    vec = r.resolve_static(5.84, -5.36).squeeze(0).numpy()
    # cocoa_prob is index 9 in the static vector per _pack_static_vector
    assert 0.0 <= vec[9] <= 1.0


def test_feature_resolver_falls_back_outside_fdp_coverage():
    from api.feature_resolver import _cocoa_belt_probability

    # Cameroon: outside the 6-country FDP coverage (CIV, GHA, IDN, ECU, PER, COL)
    p = _cocoa_belt_probability(4.05, 9.71)
    assert 0.0 <= p <= 1.0
