"""Backward-compatible shim; implementation in models.process.casej_process."""

import importlib as _importlib

_mod = _importlib.import_module("models.process.casej_process")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
