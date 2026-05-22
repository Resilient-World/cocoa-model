"""Backward-compatible shim; implementation in models.surrogate.yield_surrogate_v2_teleconnection."""

import importlib as _importlib

_mod = _importlib.import_module("models.surrogate.yield_surrogate_v2_teleconnection")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
