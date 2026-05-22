"""Backward-compatible shim; implementation in models.conformal.aci."""

import importlib as _importlib

_mod = _importlib.import_module("models.conformal.aci")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
