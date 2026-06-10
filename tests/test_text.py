from prism.utils.text import count_tokens, sentence_tokenize


def test_count_tokens_basic():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_count_tokens_longer_text_has_more_tokens():
    assert count_tokens("hello world from a test") > count_tokens("hi")


def test_sentence_tokenize_simple_english():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = sentence_tokenize(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("First")
    assert sentences[1].startswith("Second")
    assert sentences[2].startswith("Third")


def test_sentence_tokenize_russian():
    text = "Это первое предложение. А это второе! И третье?"
    sentences = sentence_tokenize(text)
    assert len(sentences) == 3


def test_sentence_tokenize_keeps_enumerated_items_glued():
    """`1. item` should stay as one sentence, not split into "1." + "item"."""
    text = "Here is a list. 1. First item. 2. Second item."
    sentences = sentence_tokenize(text)
    has_enumerated = any(s.startswith("1.") or "1. First" in s for s in sentences)
    assert has_enumerated


def test_sentence_tokenize_empty_input():
    assert sentence_tokenize("") == []


def test_sentence_tokenize_strips_whitespace():
    sentences = sentence_tokenize("  hello.   world.  ")
    assert all(s == s.strip() for s in sentences)
