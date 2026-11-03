import re

from contextlens.schemas import Turn

LOW_CONFIDENCE = 0.5


_LINE_BREAKS = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def escape(text: str) -> str:
    """Make `<`, `>` and `&` inert and keep the text on one line inside a data block."""
    flat = _LINE_BREAKS.sub(" ", text)
    return flat.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape(text: str) -> str:
    """Undo `escape` for model output that echoed escaped prompt text, even if echoed repeatedly."""
    while True:
        plain = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        if plain == text:
            return text
        text = plain


def effective_speaker(turn: Turn) -> str:
    return "unknown" if turn.speaker_confidence < LOW_CONFIDENCE else turn.speaker


def transcript_block(turns: list[Turn], with_ids: bool = False) -> str:
    """The one place transcript text is put into a prompt, escaped, inside a <transcript> block."""
    lines = [
        f"{f'[{escape(t.id)}] ' if with_ids else ''}{escape(effective_speaker(t))}: {escape(t.text)}"
        for t in turns
    ]
    return "<transcript>\n" + "\n".join(lines) + "\n</transcript>"
