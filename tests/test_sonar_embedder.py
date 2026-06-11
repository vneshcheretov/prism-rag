"""Tests for ``SonarEmbedder``.

Unit tests run everywhere with no model. The integration tests perform real
SONAR inference (first run downloads the ``text_sonar_basic_encoder``
checkpoint, ~3 GB) and are excluded by default; run them explicitly:

    pytest -m integration tests/test_sonar_embedder.py
"""

import numpy as np
import pytest

from prism.embeddings.sonar import SonarEmbedder, resolve_flores_code


def test_resolve_flores_code_short_iso():
    assert resolve_flores_code("ru") == "rus_Cyrl"
    assert resolve_flores_code("EN") == "eng_Latn"


def test_resolve_flores_code_flores_passthrough():
    assert resolve_flores_code("rus_Cyrl") == "rus_Cyrl"


def test_resolve_flores_code_unknown_passthrough():
    assert resolve_flores_code("tlh") == "tlh"


async def test_embed_empty_skips_model_load():
    embedder = SonarEmbedder()
    out = await embedder.embed([])
    assert out.shape == (0, SonarEmbedder.EMBED_DIM)
    assert out.dtype == np.float32
    assert embedder._pipeline is None


@pytest.mark.integration
class TestSonarInference:
    """Real embedding computation on example sentences."""

    @pytest.fixture(scope="class")
    def embedder(self) -> SonarEmbedder:
        pytest.importorskip("sonar")
        return SonarEmbedder(source_lang="en", device="cpu", batch_size=4)

    async def test_embed_returns_normalized_float32(self, embedder: SonarEmbedder):
        texts = [
            "The hotel has a heated outdoor pool.",
            "Breakfast is served from 7 to 10 am.",
        ]
        vecs = await embedder.embed(texts)
        assert vecs.shape == (len(texts), embedder.dim)
        assert vecs.dtype == np.float32
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    async def test_cross_lingual_similarity(self, embedder: SonarEmbedder):
        """A Russian sentence must land closer to its English translation
        than to an unrelated English sentence."""
        [en_pool] = await embedder.embed(
            ["The hotel has a heated outdoor pool."], source_lang="en"
        )
        [ru_pool] = await embedder.embed(
            ["В отеле есть подогреваемый открытый бассейн."], source_lang="ru"
        )
        [en_other] = await embedder.embed(
            ["The invoice must be paid within thirty days."], source_lang="en"
        )

        sim_translation = float(ru_pool @ en_pool)
        sim_unrelated = float(ru_pool @ en_other)
        assert sim_translation > 0.7
        assert sim_translation > sim_unrelated
