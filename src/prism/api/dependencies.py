from __future__ import annotations

from fastapi import Request

from ..core.engine import Prism


def get_prism(request: Request) -> Prism:
    return request.app.state.prism  # type: ignore[no-any-return]
