"""MarkdownStructurer: LLM-inferred headings spliced into flat text."""

from prism import MarkdownStructurer


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


async def test_structure_empty_input_returns_as_is():
    llm = HeadingLLM("```markdown\n```")
    assert await MarkdownStructurer(llm).structure("") == ""
    assert llm.calls == 0  # no chunks -> no LLM call
