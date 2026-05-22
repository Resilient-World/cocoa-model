"""Backward-compatible shim; implementation in models.surrogate.joint_exposure_yield."""

import importlib as _importlib

_mod = _importlib.import_module("models.surrogate.joint_exposure_yield")
globals().update({n: getattr(_mod, n) for n in dir(_mod) if not n.startswith("__")})
