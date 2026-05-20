"""Unit tests for cocoa region presets (no Earth Engine)."""

from data.cocoa_exposure import (
    REGIONS,
    normalize_region_key,
    processed_era5_zarr_path,
    processed_sentinel_tif_path,
    region_bounds_dict,
    region_latlon_bounds,
)


def test_region_bounds_dict_ghana() -> None:
    b = region_bounds_dict("ghana")
    assert b["west"] < b["east"]
    assert b["south"] < b["north"]


def test_region_latlon_bounds_matches_preset() -> None:
    lat_min, lat_max, lon_min, lon_max = region_latlon_bounds("civ")
    preset = REGIONS["civ"]
    assert lat_min == preset.south
    assert lat_max == preset.north
    assert lon_min == preset.west
    assert lon_max == preset.east


def test_processed_paths_are_region_tagged(tmp_path) -> None:
    era5 = processed_era5_zarr_path("indonesia", repo_root=tmp_path, start_year=2020, end_year=2024)
    assert era5.name == "era5_indonesia_2020_2024.zarr"
    s2 = processed_sentinel_tif_path("peru", repo_root=tmp_path)
    assert s2.name == "s2_s1_peru.tif"


def test_normalize_region_aliases() -> None:
    assert normalize_region_key("CMR") == "cameroon"
    assert normalize_region_key("idn") == "indonesia"
