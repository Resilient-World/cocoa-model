"""Placeholder CMIP7 ingestion entrypoint."""

from __future__ import annotations

import logging


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
    logging.warning("CMIP7 ensemble not yet published on the configured path")
    logging.warning("TODO: confirm AR7 source archive, harmonize SSP labels, build ensemble Zarr")


if __name__ == "__main__":
    main()
