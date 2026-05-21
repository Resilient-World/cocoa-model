import os

import pytest
import ee  # noqa: F401

from data.cocoa_exposure import (
    CocoaExposureIngest,
    FDP_COCOA_COLLECTION,
    GLOBAL_AEF_GAL_WEIGHTS,
    REGIONS,
    is_fdp_covered,
    normalize_region_key,
    point_in_region,
    region_latlon_bounds,
    sample_cocoa_probability_at_point,
)


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


def test_agrifm_point_inference_without_gee(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(aoi=object(), backend="agrifm")  # type: ignore[arg-type]
    monkeypatch.setattr(ing, "_agrifm_probability_at_point", lambda lat, lon: 0.33)
    p = ing.sample_point(5.84, -5.36)
    assert p == pytest.approx(0.33, abs=0.01)


def test_ensemble_v2_blends_four_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    ing = CocoaExposureIngest(
        aoi=object(),  # type: ignore[arg-type]
        backend="ensemble_v2",
        region="ghana",
    )
    monkeypatch.setattr(ing, "_aef_probability_at_point", lambda lat, lon: 0.2)
    monkeypatch.setattr(ing, "_galileo_probability_at_point", lambda lat, lon: 0.4)
    monkeypatch.setattr(ing, "_agrifm_probability_at_point", lambda lat, lon: 0.6)
    monkeypatch.setattr(ing, "_fdp_probability_at_point", lambda lat, lon, scale_m: 0.8)
    p = ing._ensemble_v2_blend(6.0, -4.0, scale_m=10)
    assert 0.0 <= p <= 1.0


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


def test_regions_include_fdp_countries():
    assert set(REGIONS) >= {
        "ghana",
        "civ",
        "cameroon",
        "nigeria",
        "indonesia",
        "ecuador",
        "peru",
        "colombia",
    }
    assert normalize_region_key("gha") == "ghana"


def test_point_in_region_ghana():
    assert point_in_region(6.0, -1.0, "ghana")
    assert not point_in_region(6.0, -1.0, "indonesia")


def test_is_fdp_covered_cameroon():
    assert is_fdp_covered(4.05, 9.71)


def test_global_fallback_outside_all_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data.cocoa_exposure.is_fdp_covered", lambda lat, lon: False)
    monkeypatch.setattr(
        "data.cocoa_exposure._global_aef_galileo_agrifm_probability",
        lambda lat, lon, **kwargs: 0.61,
    )
    p = sample_cocoa_probability_at_point(45.0, 2.0)
    assert p == pytest.approx(0.61, abs=0.01)
