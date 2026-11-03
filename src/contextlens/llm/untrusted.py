from contextlens.schemas import Turn

LOW_CONFIDENCE = 0.5


def escape(text: str) -> str:
    """Make `<`, `>` and `&` inert so untrusted text cannot close or open a data block."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def effective_speaker(turn: Turn) -> str:
    return "unknown" if turn.speaker_confidence < LOW_CONFIDENCE else turn.speaker


def transcript_block(turns: list[Turn], with_ids: bool = False) -> str:
    """The one place transcript text is put into a prompt, escaped, inside a <transcript> block."""
    lines = [
        f"{f'[{escape(t.id)}] ' if with_ids else ''}{escape(effective_speaker(t))}: {escape(t.text)}"
        for t in turns
    ]
    return "<transcript>\n" + "\n".join(lines) + "\n</transcript>"
