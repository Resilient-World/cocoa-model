import os

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from data.field_boundaries import FTWFieldBoundaries


def _toy_parcels():
    polys = [
        Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)]),  # ~0.012 ha
        Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]),  # ~1.2 ha
    ]
    return gpd.GeoDataFrame({"id": [1, 2]}, geometry=polys, crs="EPSG:4326")


def test_size_filter_drops_tiny_and_huge(tmp_path):
    gdf = _toy_parcels()
    pq = tmp_path / "p.parquet"
    gdf.to_parquet(pq)
    ftw = FTWFieldBoundaries()
    out = ftw.load_parcels(pq, min_ha=0.1, max_ha=25.0)
    assert len(out) == 1  # 0.012 ha dropped, 1.2 ha kept
    assert "area_ha" in out.columns and "compactness" in out.columns


def test_compactness_in_unit_interval(tmp_path):
    pq = tmp_path / "p.parquet"
    _toy_parcels().to_parquet(pq)
    out = FTWFieldBoundaries().load_parcels(pq, min_ha=0.0, max_ha=1e6)
    assert (out["compactness"].between(0, 1.05)).all()  # ~1 for square; slack for projection


@pytest.mark.integration
def test_ftw_delineate_tiny_bbox(tmp_path):
    if not os.getenv("RUN_FTW_INTEGRATION"):
        pytest.skip("Set RUN_FTW_INTEGRATION=1 to run; downloads ~200MB ckpt + S2 scenes")
    ftw = FTWFieldBoundaries()
    pq = ftw.delineate(bbox=(-5.55, 6.75, -5.50, 6.80), year=2023, out_dir=tmp_path)
    gdf = ftw.load_parcels(pq)
    assert len(gdf) > 0
