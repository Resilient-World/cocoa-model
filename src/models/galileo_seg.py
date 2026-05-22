"""Backward-compatible shim; implementation in models.backbones.galileo_seg."""

import importlib as _importlib

_mod = _importlib.import_module("models.backbones.galileo_seg")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
