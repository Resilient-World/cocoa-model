"""Backward-compatible shim; implementation in models.conformal.validate_conformal_coverage."""

import importlib as _importlib

_mod = _importlib.import_module("models.conformal.validate_conformal_coverage")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
