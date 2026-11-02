from typing import Protocol

from contextlens.schemas import Turn


class TranscriptStore(Protocol):
    def add(self, turn: Turn) -> None: ...

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]: ...

    def delete(self, conv_id: str) -> None: ...


class InMemoryTranscriptStore:
    def __init__(self) -> None:
        self._by_conv: dict[str, list[Turn]] = {}

    def add(self, turn: Turn) -> None:
        self._by_conv.setdefault(turn.conversation_id, []).append(turn)

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]:
        found = [t for t in self._by_conv.get(conv_id, []) if start_seq <= t.seq <= as_of_seq]
        return sorted(found, key=lambda t: t.seq)

    def delete(self, conv_id: str) -> None:
        self._by_conv.pop(conv_id, None)
