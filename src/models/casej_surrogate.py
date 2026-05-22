"""Backward-compatible shim; implementation in models.process.casej_surrogate."""

import importlib as _importlib

_mod = _importlib.import_module("models.process.casej_surrogate")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
