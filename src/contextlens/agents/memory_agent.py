from contextlens.agents.specialist import system_prompt
from contextlens.prompts.registry import get_prompt

NAME, KIND = "memory", "recall"
TOOLS = ["search_transcript", "get_turns", "lookup_memory"]
_ROLE = (
    "You are the memory agent. Find earlier information in the conversation that is relevant to the question, "
    "and who said it. Name a speaker only as the transcript shows them."
)


def system(prompt_version: str) -> str:
    return system_prompt(_ROLE, get_prompt("recall", prompt_version).text, NAME, KIND, TOOLS)
