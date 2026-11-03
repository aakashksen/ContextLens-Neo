import re
from pathlib import Path

from pydantic import BaseModel

from contextlens.llm.untrusted import transcript_block
from contextlens.schemas import Turn

_DIR = Path(__file__).parent
_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class Prompt(BaseModel):
    name: str
    version: str
    text: str


def get_prompt(name: str, version: str = "v1") -> Prompt:
    for label, value in (("name", name), ("version", version)):
        if not _PLAIN.fullmatch(value):
            raise ValueError(f"prompt {label} must be a plain word, got {value!r}")
    path = (_DIR / version / f"{name}.md").resolve()
    if path.parent.parent != _DIR.resolve() or not path.is_file():
        raise KeyError(f"no prompt {name!r} at version {version!r}")
    return Prompt(
        name=name, version=version, text=path.read_text(encoding="utf-8").strip()
    )


def render_transcript_block(turns: list[Turn]) -> str:
    """Turns as an escaped data block with ids; text inside cannot close the block."""
    return transcript_block(turns, with_ids=True)
