from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """API runtime configuration, read from environment variables."""

    qdrant_url: str
    collection_name: str
    sonar_device: str | None
    language: str | None
    recreate_collection: bool

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("PRISM_COLLECTION", "prism"),
            sonar_device=os.getenv("PRISM_SONAR_DEVICE"),
            language=os.getenv("PRISM_LANGUAGE"),
            recreate_collection=os.getenv("PRISM_RECREATE_COLLECTION", "false").lower()
            == "true",
        )
