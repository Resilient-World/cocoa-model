"""Backward-compatible shim; implementation in models.surrogate.ensemble_surrogate."""

import importlib as _importlib

_mod = _importlib.import_module("models.surrogate.ensemble_surrogate")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
