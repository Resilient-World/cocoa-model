"""Conformal prediction subpackage."""

import importlib as _importlib

_mod = _importlib.import_module("models.conformal.conformal")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})

# Explicit exports for type checkers
from models.conformal.conformal import (
    ConformalInterval,
    ConformalPredictor,
    MondrianConformalYield,
    SplitConformalYield,
    load_conformal,
    load_conformal_if_exists,
    save_conformal,
)

__all__ = [
    "ConformalInterval",
    "ConformalPredictor",
    "MondrianConformalYield",
    "SplitConformalYield",
    "load_conformal",
    "load_conformal_if_exists",
    "save_conformal",
]
