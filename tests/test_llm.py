"""LLMClient token-usage accounting."""

from types import SimpleNamespace

from pydantic import BaseModel

from prism import LLMClient


class _Schema(BaseModel):
    x: int


class FakeOpenAI:
    """Minimal stand-in for AsyncOpenAI returning canned usage."""

    def __init__(self) -> None:
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse))
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _parse(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=_Schema(x=1), refusal=None))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


async def test_token_usage_accumulates_across_calls():
    llm = LLMClient(client=FakeOpenAI())
    assert llm.usage.total_tokens == 0

    await llm.complete_structured("sys", "user", _Schema)
    await llm.complete_text("sys", "user")

    assert llm.usage.calls == 2
    assert llm.usage.prompt_tokens == 13
    assert llm.usage.completion_tokens == 7
    assert llm.usage.total_tokens == 20


async def test_token_usage_reset():
    llm = LLMClient(client=FakeOpenAI())
    await llm.complete_text("sys", "user")
    assert llm.usage.calls == 1
    llm.usage.reset()
    assert llm.usage.calls == 0
    assert llm.usage.total_tokens == 0
