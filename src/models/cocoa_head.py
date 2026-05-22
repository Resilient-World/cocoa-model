"""Backward-compatible shim; implementation in models.surrogate.cocoa_head."""

import importlib as _importlib

_mod = _importlib.import_module("models.surrogate.cocoa_head")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
