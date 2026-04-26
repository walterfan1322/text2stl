"""Backend plugin system for 3D modeling engines (trimesh, CadQuery...)."""
from __future__ import annotations

from .base import ModelBackend, BackendError

_REGISTRY: dict[str, ModelBackend] = {}


def register(name: str, backend: ModelBackend) -> None:
    _REGISTRY[name] = backend


def get_backend(name: str) -> ModelBackend:
    if name not in _REGISTRY:
        raise BackendError(
            f"Unknown backend: {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_backends() -> list[str]:
    return list(_REGISTRY)


# Lazy-register built-in backends. Each backend module self-registers on import.
# We import them here so `from backends import get_backend` just works.
from . import trimesh_backend  # noqa: E402,F401
from . import cadquery_backend  # noqa: E402,F401
