from prism.utils.lexical import (
    chunk_lexical_payload,
    keypoint_stems,
    sentence_contains_all,
    stem_tokens,
)


def test_stem_tokens_russian_inflections_collapse():
    """Different surface forms of the same lemma share a stem."""
    a = stem_tokens("новая машина")
    b = stem_tokens("новые машины")
    c = stem_tokens("новой машине")
    assert a == b == c == ["нов", "машин"]


def test_stem_tokens_distinguishes_homograph_roots():
    """`новый` and `новость` share a prefix but are different lemmas —
    a good stemmer must not collapse them, otherwise the AND filter
    would over-match queries like 'новая машина' against 'новости о машинах'.
    """
    assert "нов" in stem_tokens("новая")
    assert "нов" not in stem_tokens("новости")
    assert "новост" in stem_tokens("новости")


def test_stem_tokens_drops_stopwords_and_punctuation():
    stems = stem_tokens("Это новая машина, и она едет по дороге.")
    assert "это" not in stems
    assert "и" not in stems
    assert "по" not in stems
    assert "нов" in stems
    assert "машин" in stems


def test_stem_tokens_handles_empty_and_whitespace():
    assert stem_tokens("") == []
    assert stem_tokens("   \n  ") == []


def test_keypoint_stems_deduplicates_preserving_order():
    """Multiple occurrences of the same word collapse to one stem in
    occurrence order, so query and index sides agree on ordering."""
    stems = keypoint_stems("машина машина новая машина")
    assert stems == ["машин", "нов"]


def test_chunk_lexical_payload_basic_structure():
    text = "Новая машина едет. Она красная и яркая. Машина тихая."
    payload = chunk_lexical_payload(text)

    assert set(payload["stems"]) >= {"нов", "машин", "едет", "красн", "ярк", "тих"}
    assert isinstance(payload["sentence_stems"], list)
    assert len(payload["sentence_stems"]) == 3
    assert all(isinstance(s, list) for s in payload["sentence_stems"])


def test_chunk_lexical_payload_sentence_stems_are_unique_within_sentence():
    text = "Машина машина машина едет."
    payload = chunk_lexical_payload(text)
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
    payload = chunk_lexical_payload(text)
    query = keypoint_stems("новая машина")

    # Stem-level AND would have passed (both stems present in the chunk)
    assert set(query).issubset(set(payload["stems"]))
    # Sentence-level proximity correctly rejects
    assert sentence_contains_all(payload["sentence_stems"], query) is False


def test_end_to_end_target_chunk_is_accepted():
    """A chunk where 'нов' and 'машин' co-occur in one sentence should pass."""
    text = "Я купил новые красные машины вчера. Они стоят в гараже."
    payload = chunk_lexical_payload(text)
    query = keypoint_stems("новая машина")

    assert set(query).issubset(set(payload["stems"]))
    assert sentence_contains_all(payload["sentence_stems"], query) is True


def test_end_to_end_news_chunk_is_rejected_via_different_stem():
    """Query 'новая машина' vs chunk 'новости о машинах' — the Russian
    Snowball stemmer treats 'новости' (новост) and 'новая' (нов) as
    different lemmas, so AND already rejects without needing post-filter.
    """
    text = "Свежие новости о машинах появились на сайте сегодня утром."
    payload = chunk_lexical_payload(text)
    query = keypoint_stems("новая машина")

    assert "нов" not in payload["stems"]
    assert "новост" in payload["stems"]
    assert not set(query).issubset(set(payload["stems"]))
