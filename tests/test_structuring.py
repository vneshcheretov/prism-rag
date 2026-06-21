"""MarkdownStructurer: LLM-inferred headings spliced into flat text."""

from prism import MarkdownStructurer
from prism.schemas.llm_outputs import DocumentTitle


class HeadingLLM:
    """Returns canned numbered headings in a ```markdown fence."""

    def __init__(self, fence: str) -> None:
        self.fence = fence
        self.calls = 0

    async def complete_structured(self, *args, **kwargs):
        raise AssertionError("structurer should only use complete_text")

    async def complete_text(self, system, user, *, tier="strong"):
        self.calls += 1
        return self.fence


async def test_structure_splices_headings_into_text():
    llm = HeadingLLM("```markdown\n1) # Hotel\n2) ## Pets\n```")
    data = "Aiso hotel overview text. Pets up to five kg are allowed."

    result = await MarkdownStructurer(llm).structure(data)

    assert "# Hotel" in result
    assert "## Pets" in result
    # original sentence content is preserved, not replaced
    assert "Pets up to five kg are allowed" in result
    assert llm.calls == 1


async def test_structure_tolerates_missing_markdown_fence():
    # a model that returns headings without the ```markdown fence must still work
    llm = HeadingLLM("1) # Hotel\n2) ## Pets")
    data = "Aiso hotel overview text. Pets up to five kg are allowed."
    result = await MarkdownStructurer(llm).structure(data)
    assert "# Hotel" in result
    assert "## Pets" in result


async def test_structure_puts_heading_on_its_own_line():
    # heading must not glue to the following sentence (chunker parses by line)
    llm = HeadingLLM("```markdown\n2) ## Pets\n```")
    data = "Aiso hotel overview text. Pets up to five kg are allowed."
    result = await MarkdownStructurer(llm).structure(data)
    assert "## Pets\nPets up to five kg" in result


async def test_structure_empty_input_returns_as_is():
    llm = HeadingLLM("```markdown\n```")
    assert await MarkdownStructurer(llm).structure("") == ""
    assert llm.calls == 0  # no chunks -> no LLM call


class TitleLLM:
    """Returns a canned structured title; rejects text calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(self, system, user, schema, *, tier="fast"):
        self.calls += 1
        return DocumentTitle(title="Hotel Handbook")

    async def complete_text(self, *args, **kwargs):
        raise AssertionError("generate_title should use complete_structured")


async def test_generate_title_from_first_chunk():
    llm = TitleLLM()
    title = await MarkdownStructurer(llm).generate_title("Aiso hotel overview. Pets allowed.")
    assert title == "Hotel Handbook"
    assert llm.calls == 1


async def test_generate_title_empty_returns_none():
    llm = TitleLLM()
    assert await MarkdownStructurer(llm).generate_title("   ") is None
    assert llm.calls == 0  # no sentences -> no LLM call
