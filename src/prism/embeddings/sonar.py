from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

log = logging.getLogger(__name__)


# Common ISO 639-1 codes → FLORES-200 codes that SONAR expects.
# Covers the ~12 languages most users will reach for; anything not here is
# passed through as-is so power users can still use raw FLORES codes.
ISO_TO_FLORES: dict[str, str] = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ja": "jpn_Jpan",
    "kk": "kaz_Cyrl",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
}


def resolve_flores_code(lang: str) -> str:
    """Resolve a short ISO code (``"ru"``) to a FLORES-200 code (``"rus_Cyrl"``).

    A code already in FLORES form (``"<lang>_<script>"``) is returned unchanged.
    An unknown short code is also returned unchanged with a warning — SONAR
    will raise a clearer error than this helper could.
    """
    if "_" in lang:
        return lang
    key = lang.lower()
    if key in ISO_TO_FLORES:
        return ISO_TO_FLORES[key]
    log.warning("unknown language code %r; passing through to SONAR as-is", lang)
    return lang


class SonarEmbedder(Embedder):
    """Meta SONAR text embedder.

    Wraps ``sonar.inference_pipelines.text.TextToEmbeddingModelPipeline`` with
    a small layer that fits Prism's needs:

    - lazy model load (no torch import until first ``embed`` call)
    - automatic CUDA detection
    - async ``embed`` that off-loads the synchronous torch inference to a
      worker thread so the event loop is not blocked
    - L2 normalization at the embedder boundary, so the rest of the pipeline
      can treat cosine similarity as a plain dot product

    The pretrained encoder ``text_sonar_basic_encoder`` produces 1024-dim
    embeddings and covers 200 languages via FLORES-200 codes (e.g.
    ``eng_Latn``, ``rus_Cyrl``).
    """

    EMBED_DIM = 1024
    ENCODER_NAME = "text_sonar_basic_encoder"

    def __init__(
        self,
        source_lang: str = "eng_Latn",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.source_lang = resolve_flores_code(source_lang)
        self._device_override = device
        self.batch_size = batch_size
        self._pipeline: TextToEmbeddingModelPipeline | None = None
        self._resolved_device: str | None = None

    @property
    def dim(self) -> int:
        return self.EMBED_DIM

    @property
    def device(self) -> str:
        return self._resolved_device or self._resolve_device()

    def _resolve_device(self) -> str:
        if self._device_override is not None:
            return self._device_override
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        try:
            import torch
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        except ImportError as e:
            raise ImportError(
                "SONAR dependencies are not installed. "
                "Install the optional extra: `pip install 'prism-rag[sonar]'`."
            ) from e

        self._resolved_device = self._resolve_device()
        log.debug("Loading SONAR %s on %s ...", self.ENCODER_NAME, self._resolved_device)
        self._pipeline = TextToEmbeddingModelPipeline(
            encoder=self.ENCODER_NAME,
            tokenizer=self.ENCODER_NAME,
            device=torch.device(self._resolved_device),
        )
        log.debug("SONAR loaded.")

    def _embed_sync(self, texts: list[str], source_lang: str) -> np.ndarray:
        self._load()
        assert self._pipeline is not None

        tensor: Any = self._pipeline.predict(
            texts,
            source_lang=source_lang,
            batch_size=self.batch_size,
        )
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    async def embed(
        self,
        texts: list[str],
        source_lang: str | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)

        lang = resolve_flores_code(source_lang) if source_lang else self.source_lang
        return await asyncio.to_thread(self._embed_sync, texts, lang)
