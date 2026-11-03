from contextlens.agents.specialist import system_prompt
from contextlens.prompts.registry import get_prompt

NAME, KIND = "consistency", "consistency"
_ROLE = (
    "You are the consistency agent. Decide whether the latest statement contradicts earlier turns, is a "
    "legitimate update, agrees with them, or whether there is too little evidence. A change someone announces "
    '("we moved it", "actually, make it") is an update, not a contradiction. Put the decision in the finding\'s "label".'
)


def tool_names(has_checker: bool) -> list[str]:
    return ["search_transcript", *["check_consistency"] * has_checker]


def system(prompt_version: str, has_checker: bool) -> str:
    return system_prompt(_ROLE, get_prompt("recall", prompt_version).text, NAME, KIND, tool_names(has_checker))
