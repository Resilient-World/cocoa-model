"""Pytest configuration: repo root on ``sys.path`` for ``scripts`` imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

API_KEY_HEADERS = {"x-api-key": "test-cocoa-key"}


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCOA_MODEL_API_KEY", "test-cocoa-key")
