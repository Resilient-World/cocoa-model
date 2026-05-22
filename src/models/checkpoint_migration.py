"""Backward-compatible shim; implementation in models.io.checkpoint_migration."""
import importlib as _importlib

_mod = _importlib.import_module("models.io.checkpoint_migration")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
