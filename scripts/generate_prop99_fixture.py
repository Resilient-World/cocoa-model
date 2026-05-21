#!/usr/bin/env python3
"""Refresh California Prop 99 fixture from synth-inference/synthdid."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthdid" / "california_prop99.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
