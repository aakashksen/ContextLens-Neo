from contextlens.schemas import Scenario


def query_for(sc: Scenario) -> tuple[str, int]:
    """The query text and the last seq that may be searched."""
    if (sc.request.question or "").strip():
        return sc.request.question, sc.request.as_of_seq
    last = max((t for t in sc.conversation.turns if t.seq <= sc.request.as_of_seq), key=lambda t: t.seq)
    return last.text, sc.request.as_of_seq - 1
