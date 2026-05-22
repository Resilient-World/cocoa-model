"""Backward-compatible shim; implementation in models.features.pape."""
import importlib as _importlib

_mod = _importlib.import_module("models.features.pape")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
