"""Backward-compatible shim; implementation in models.backbones.terramind_backbone."""

import importlib as _importlib

_mod = _importlib.import_module("models.backbones.terramind_backbone")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
