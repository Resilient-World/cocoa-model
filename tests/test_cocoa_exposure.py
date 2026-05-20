import os

import pytest
import ee  # noqa: F401

from data.cocoa_exposure import CocoaExposureIngest, FDP_COCOA_COLLECTION


def test_collection_id_is_fdp_2025a():
    assert FDP_COCOA_COLLECTION == "projects/forestdatapartnership/assets/cocoa/model_2025a"


def test_threshold_default_matches_kalischek_f1_optimal():
    # AOI is only stored until a GEE method runs; avoid ee.Geometry (requires Initialize).
    ing = CocoaExposureIngest(aoi=object())  # type: ignore[arg-type]
    assert ing.threshold == pytest.approx(0.65)
    assert ing.year == 2023


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
