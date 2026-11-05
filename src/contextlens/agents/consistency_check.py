from typing import Protocol

from pydantic import BaseModel

from contextlens.guard.instructions import without_instruction_turns
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
    def __init__(self, llm: LLMClient, strip_instructions: bool = False) -> None:
        """`strip_instructions`: earlier turns the instruction guard flags are left out of the prompt. Off by default;
        whether the product uses it is still to be decided."""
        self.llm = llm
        self.strip_instructions = strip_instructions
        self.prompt = get_prompt("consistency", "v1")
        self.last_tokens: int | None = None  # prompt plus completion tokens of the last call; None if it failed

    def check(
        self, statement: Turn, earlier: list[Turn]
    ) -> tuple[ConsistencyLabel, float]:
        if self.strip_instructions:
            earlier = without_instruction_turns(earlier)
        messages = [
            {"role": "system", "content": self.prompt.text},
            {"role": "user", "content": render_transcript_block([*earlier, statement])},
        ]
        self.last_tokens = None
        res = self.llm.complete(messages, schema=_Out, max_tokens=100)
        out = res.parsed
        assert isinstance(out, _Out)
        self.last_tokens = res.prompt_tokens + res.completion_tokens
        return out.label, max(0.0, min(1.0, out.confidence))
