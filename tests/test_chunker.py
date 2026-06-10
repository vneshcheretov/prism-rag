from prism.core.chunker import Chunk, MarkdownChunker


def test_chunker_produces_chunks_from_markdown():
    md = """# Intro

This is the introduction. It has several sentences. They explain the topic in detail and provide context that the reader needs to understand before moving on.

## Section A

Content of section A. More content here. And yet more sentences to bulk it up past the minimum token threshold so that the chunker actually emits a chunk for this section.

## Section B

Section B has its own content. Another sentence. And another one to make sure we cross the minimum token threshold for this section as well.
"""
    chunker = MarkdownChunker(max_tokens=512, min_section_tokens=10)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunks_carry_header_hierarchy():
    md = """# Top

## Sub

This section has enough text to survive the minimum token filter. It needs to be longer than twenty tokens. Let me keep typing until we hit that bar comfortably.
"""
    chunks = MarkdownChunker(min_section_tokens=5).chunk(md)
    assert chunks, "expected at least one chunk"
    sub_chunks = [c for c in chunks if "Sub" in c.headers]
    assert sub_chunks
    assert sub_chunks[0].headers == ["Top", "Sub"]
    # The chunk text starts with the full header chain
    assert sub_chunks[0].text.startswith("# Top\n## Sub")


def test_chunks_from_same_section_share_paragraph_id():
    sentence = "This sentence repeats. " * 80
    md = f"# H\n\n{sentence}"
    chunker = MarkdownChunker(max_tokens=80, min_section_tokens=5)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 2, "expected the section to be split into multiple chunks"
    pids = {c.paragraph_id for c in chunks}
    assert len(pids) == 1, "all chunks of one section must share a paragraph_id"


def test_short_section_is_filtered_out():
    md = """# Tiny

Two words.
"""
    chunks = MarkdownChunker(min_section_tokens=20).chunk(md)
    assert chunks == []


def test_chunker_respects_max_tokens():
    long_sentence = "The quick brown fox jumps over the lazy dog. " * 50
    md = f"# Long\n\n{long_sentence}"
    chunker = MarkdownChunker(max_tokens=64, min_section_tokens=5)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 2
    # Header chain is included in chunk.text and so counts against tokens, so we
    # allow some slack — but every chunk should be reasonably close to the budget.
    from prism.utils.text import count_tokens
    for c in chunks:
        assert count_tokens(c.text) <= 64 * 2.5, (
            f"chunk grew far past max_tokens budget: {count_tokens(c.text)}"
        )


def test_header_path_property():
    chunk = Chunk(text="x", paragraph_id="p", headers=["A", "B", "C"])
    assert chunk.header_path == "A > B > C"
