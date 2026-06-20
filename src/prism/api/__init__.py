"""FastAPI service exposing the Prism engine over HTTP."""

from .app import create_app

__all__ = ["create_app"]
