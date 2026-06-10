"""``indx app`` — INDX, the local FastAPI server for the journey UI (docs/app-spec.md).

This package is shipped as the optional ``indx[app]`` extra. ``fastapi`` / ``uvicorn`` /
``starlette`` are imported **lazily inside functions** (never at module top), exactly like
every adapter, so ``import indx.app`` stays safe on a light, core-only install. The public
names :func:`create_app` and :func:`serve` are re-exported lazily via ``__getattr__`` so that
merely importing this package never drags in the web stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, not executed at runtime
    from indx.app.server import create_app, serve

__all__ = ["create_app", "serve"]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``create_app`` / ``serve`` so ``import indx.app`` stays fastapi-free."""
    if name in __all__:
        from indx.app import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
