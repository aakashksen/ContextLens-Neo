from typing import Protocol

from pydantic import BaseModel

from contextlens.llm.client import LLMClient
from contextlens.prompts.registry import get_prompt, render_transcript_block
from contextlens.schemas import ConsistencyLabel, Turn


class ConsistencyChecker(Protocol):
    def check(
        self, statement: Turn, earlier: list[Turn]
    ) -> tuple[ConsistencyLabel, float]: ...


class _Out(BaseModel):
    label: ConsistencyLabel
    confidence: float


class PromptedChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = get_prompt("consistency", "v1")

    def check(
        self, statement: Turn, earlier: list[Turn]
    ) -> tuple[ConsistencyLabel, float]:
        messages = [
            {"role": "system", "content": self.prompt.text},
            {"role": "user", "content": render_transcript_block([*earlier, statement])},
        ]
        out = self.llm.complete(messages, schema=_Out, max_tokens=100).parsed
        assert isinstance(out, _Out)
        return out.label, max(0.0, min(1.0, out.confidence))
