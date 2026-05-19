"""Unit tests for cocoa dataset helpers (no raster I/O)."""

from pathlib import Path

import pytest

from data.cocoa_dataset import (
    CLASS_AGROFORESTRY_COCOA,
    CLASS_FULL_SUN_COCOA,
    CLASS_NAMES,
    CLASS_OTHER,
    _verify_paired_tiles,
)


def test_class_names() -> None:
    assert CLASS_NAMES[CLASS_OTHER] == "other"
    assert CLASS_NAMES[CLASS_FULL_SUN_COCOA] == "full_sun_cocoa"
    assert CLASS_NAMES[CLASS_AGROFORESTRY_COCOA] == "agroforestry_cocoa"
    assert len(CLASS_NAMES) == 3


def test_verify_paired_tiles_success(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "tile_a.tif").touch()
    (mask_dir / "tile_a.tif").touch()
    _verify_paired_tiles(image_dir, mask_dir)


def test_verify_paired_tiles_missing_mask(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "tile_a.tif").touch()
    with pytest.raises(FileNotFoundError, match="Missing mask"):
        _verify_paired_tiles(image_dir, mask_dir)
