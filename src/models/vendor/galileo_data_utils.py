"""Backward-compatible shim; implementation in models.backbones.vendor.galileo_data_utils."""

import importlib as _importlib

_mod = _importlib.import_module("models.backbones.vendor.galileo_data_utils")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
