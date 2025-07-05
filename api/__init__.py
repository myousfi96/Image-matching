"""API package initializer.

Provides lazy access to the FastAPI `app` object so that importing sub-modules
like `api.vector_db` does **not** pull in heavy FastAPI dependencies.  This
avoids requiring FastAPI in environments (e.g. Triton-server) that only need
utility modules.

When the attribute `app` is first accessed we import `api.main` and return its
`app` object.  All other attribute accesses are delegated to that module once
it is loaded.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

_main: ModuleType | None = None  # cache for api.main


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazily import ``api.main`` when `app` is requested.

    This allows ``import api.vector_db`` without installing FastAPI. When
    Uvicorn runs with ``--app-dir api:app`` it accesses the `app` attribute,
    which triggers this function and performs the import.
    """
    global _main

    if name == "app":
        if _main is None:
            _main = import_module("api.main")
        return getattr(_main, "app")

    # Fallback – if `api.main` has been imported, try to delegate other
    # attribute access to it (helpful for type-checking / re-exports).
    if _main is not None and hasattr(_main, name):
        return getattr(_main, name)

    raise AttributeError(f"module 'api' has no attribute {name!r}")

__all__ = ["app"] 