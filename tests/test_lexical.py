import pytest

from prism.schemas.llm_outputs import LLMStems
from prism.utils.lexical import (
    chunk_lexical_payload,
    chunk_lexical_payload_llm,
    keypoint_stems,
    keypoint_stems_llm,
    sentence_contains_all,
    stem_tokens,
)


class _StaticLLM:
    """Minimal async LLMClient stub.

    Returns stems by looking up each input word in ``mapping`` (defaults to
    the lowercased word if not mapped). Records ``calls`` so tests can
    assert the prompt path was actually exercised. ``fail_after`` lets a
    test simulate a transient error after N successful calls.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        *,
        fail: bool = False,
        truncate_to: int | None = None,
    ) -> None:
        self.mapping = mapping or {}
        self.fail = fail
        self.truncate_to = truncate_to
        self.calls: list[list[str]] = []

    async def complete_structured(self, *, system: str, user: str, schema, tier: str):
        import json
        words = json.loads(user)
        self.calls.append(words)
        if self.fail:
            raise RuntimeError("simulated llm failure")
        stems = [self.mapping.get(w, w) for w in words]
        if self.truncate_to is not None:
            stems = stems[: self.truncate_to]
        return LLMStems(stems=stems)


def test_stem_tokens_russian_inflections_collapse():
    """Different surface forms of the same lemma share a stem."""
    a = stem_tokens("новая машина", lang="ru")
    b = stem_tokens("новые машины", lang="ru")
    c = stem_tokens("новой машине", lang="ru")
    assert a == b == c == ["нов", "машин"]


def test_stem_tokens_distinguishes_homograph_roots():
    """`новый` and `новость` share a prefix but are different lemmas —
    a good stemmer must not collapse them, otherwise the AND filter
    would over-match queries like 'новая машина' against 'новости о машинах'.
    """
    assert "нов" in stem_tokens("новая", lang="ru")
    assert "нов" not in stem_tokens("новости", lang="ru")
    assert "новост" in stem_tokens("новости", lang="ru")


def test_stem_tokens_drops_punctuation_and_single_letter_tokens():
    """No stopword filter, but `len > 1` and the word regex still drop
    punctuation, whitespace, and one-letter noise."""
    stems = stem_tokens("Это новая машина, и она едет по дороге.", lang="ru")
    assert "," not in stems
    assert "." not in stems
    assert "и" not in stems  # single-letter — dropped by len > 1
    assert "нов" in stems
    assert "машин" in stems


def test_stem_tokens_identity_fallback_for_unsupported_language():
    """Languages outside Snowball (kk, uk, ja...) fall back to identity:
    words pass through lowercased, no morphological folding."""
    stems = stem_tokens("Жаңа көлік", lang="kk")
    assert stems == ["жаңа", "көлік"]


def test_stem_tokens_handles_empty_and_whitespace():
    assert stem_tokens("") == []
    assert stem_tokens("   \n  ") == []


def test_keypoint_stems_deduplicates_preserving_order():
    """Multiple occurrences of the same word collapse to one stem in
    occurrence order, so query and index sides agree on ordering."""
    stems = keypoint_stems("машина машина новая машина", lang="ru")
    assert stems == ["машин", "нов"]


def test_chunk_lexical_payload_basic_structure():
    text = "Новая машина едет. Она красная и яркая. Машина тихая."
    payload = chunk_lexical_payload(text, lang="ru")

    assert set(payload["stems"]) >= {"нов", "машин", "едет", "красн", "ярк", "тих"}
    assert isinstance(payload["sentence_stems"], list)
    assert len(payload["sentence_stems"]) == 3
    assert all(isinstance(s, list) for s in payload["sentence_stems"])


def test_chunk_lexical_payload_sentence_stems_are_unique_within_sentence():
    text = "Машина машина машина едет."
    payload = chunk_lexical_payload(text, lang="ru")
    sent = payload["sentence_stems"][0]
    assert sent.count("машин") == 1


def test_chunk_lexical_payload_empty_text():
    payload = chunk_lexical_payload("")
    assert payload["stems"] == []
    assert payload["sentence_stems"] == []


def test_sentence_contains_all_positive():
    sentences = [["нов", "машин", "едет"], ["красн", "ярк"]]
    assert sentence_contains_all(sentences, ["нов", "машин"]) is True


def test_sentence_contains_all_requires_co_occurrence_in_single_sentence():
    """The whole point of flavor S — stems must be in the same sentence.
    Splitting them across sentences must NOT pass the filter.
    """
    sentences = [["нов", "закон"], ["машин", "врем"]]
    assert sentence_contains_all(sentences, ["нов", "машин"]) is False


def test_sentence_contains_all_subset_match():
    """Sentence may have extra stems beyond the required ones — still a match."""
    sentences = [["нов", "красн", "машин", "едет", "дорог"]]
    assert sentence_contains_all(sentences, ["нов", "машин"]) is True


def test_sentence_contains_all_empty_required_is_vacuously_true():
    assert sentence_contains_all([["x"]], []) is True


def test_sentence_contains_all_empty_sentences_is_false():
    assert sentence_contains_all([], ["нов"]) is False


def test_end_to_end_hostile_chunk_is_rejected():
    """The motivating example: 'нов' and 'машин' appear in the same chunk
    but in different sentences. AND-on-stems would let it through; flavor S
    must reject it.
    """
    text = (
        "Новый закон был принят парламентом. "
        "После долгого обсуждения. "
        "Машина времени — известный концепт в фантастике."
    )
    payload = chunk_lexical_payload(text, lang="ru")
    query = keypoint_stems("новая машина", lang="ru")

    # Stem-level AND would have passed (both stems present in the chunk)
    assert set(query).issubset(set(payload["stems"]))
    # Sentence-level proximity correctly rejects
    assert sentence_contains_all(payload["sentence_stems"], query) is False


def test_end_to_end_target_chunk_is_accepted():
    """A chunk where 'нов' and 'машин' co-occur in one sentence should pass."""
    text = "Я купил новые красные машины вчера. Они стоят в гараже."
    payload = chunk_lexical_payload(text, lang="ru")
    query = keypoint_stems("новая машина", lang="ru")

    assert set(query).issubset(set(payload["stems"]))
    assert sentence_contains_all(payload["sentence_stems"], query) is True


def test_end_to_end_news_chunk_is_rejected_via_different_stem():
    """Query 'новая машина' vs chunk 'новости о машинах' — the Russian
    Snowball stemmer treats 'новости' (новост) and 'новая' (нов) as
    different lemmas, so AND already rejects without needing post-filter.
    """
    text = "Свежие новости о машинах появились на сайте сегодня утром."
    payload = chunk_lexical_payload(text, lang="ru")
    query = keypoint_stems("новая машина", lang="ru")

    assert "нов" not in payload["stems"]
    assert "новост" in payload["stems"]
    assert not set(query).issubset(set(payload["stems"]))


# --- LLM stemmer fallback ----------------------------------------------------


@pytest.mark.asyncio
async def test_llm_keypoint_stems_kazakh_collapses_inflections():
    """Kazakh has no Snowball algorithm — the LLM stemmer should collapse
    case suffixes (-тер, -де) to their base lemma so index and query agree.
    """
    llm = _StaticLLM(mapping={
        "көліктер": "көлік",
        "көлікте": "көлік",
        "жаңа": "жаңа",
    })
    stems = await keypoint_stems_llm("жаңа көліктер", llm, "Kazakh")
    assert stems == ["жаңа", "көлік"]
    assert llm.calls == [["жаңа", "көліктер"]]


@pytest.mark.asyncio
async def test_llm_chunk_payload_preserves_sentence_structure():
    """Per-sentence stem lists are rebuilt from the flat LLM response."""
    llm = _StaticLLM(mapping={"көліктер": "көлік", "жаңа": "жаңа", "көлікте": "көлік"})
    text = "Жаңа көліктер келді. Көлікте отыр."
    payload = await chunk_lexical_payload_llm(text, llm, "Kazakh")

    assert set(payload["stems"]) >= {"жаңа", "көлік"}
    assert len(payload["sentence_stems"]) == 2
    # second sentence's only content stem is "көлік"
    assert "көлік" in payload["sentence_stems"][1]


@pytest.mark.asyncio
async def test_llm_stemmer_falls_back_to_identity_on_error():
    """When the LLM call raises, we keep going with the raw words rather
    than failing ingest — partial recall is better than a broken pipeline."""
    llm = _StaticLLM(fail=True)
    stems = await keypoint_stems_llm("жаңа көліктер", llm, "Kazakh")
    # identity-ish: lowercased words pass through, both have len > 1
    assert stems == ["жаңа", "көліктер"]


@pytest.mark.asyncio
async def test_llm_stemmer_falls_back_on_length_mismatch():
    """A truncated LLM response can't be mapped back to per-word positions
    safely, so it's treated the same as a failure."""
    llm = _StaticLLM(
        mapping={"жаңа": "жаңа", "көліктер": "көлік"},
        truncate_to=1,
    )
    stems = await keypoint_stems_llm("жаңа көліктер", llm, "Kazakh")
    assert stems == ["жаңа", "көліктер"]
