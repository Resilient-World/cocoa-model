"""Backward-compatible shim; implementation in models.process.case2_runner."""

import importlib as _importlib

_mod = _importlib.import_module("models.process.case2_runner")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
