"""Backward-compatible shim; implementation in models.backbones.vendor.single_file_galileo."""

import importlib as _importlib

_mod = _importlib.import_module("models.backbones.vendor.single_file_galileo")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
