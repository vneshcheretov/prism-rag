"""Run the Prism API with uvicorn: ``python -m prism.api``."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "prism.api.app:create_app",
        factory=True,
        host=os.getenv("PRISM_API_HOST", "0.0.0.0"),
        port=int(os.getenv("PRISM_API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
