from contextlens.agents.consistency_check import ConsistencyChecker
from contextlens.guard.citations import resolve_turn
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import TranscriptStore
from contextlens.retrieval.hybrid import Retriever, ScoredTurn
from contextlens.schemas import ConsistencyLabel, MemoryItem, Turn


class Tools:
    """What an agent may read. Nothing here returns a turn after `as_of_seq`."""

    def __init__(
        self,
        store: TranscriptStore,
        retriever: Retriever,
        memory: MemoryStore,
        as_of_seq: int,
        conv_id: str,
        checker: ConsistencyChecker | None = None,
    ) -> None:
        self.store, self.retriever, self.memory = store, retriever, memory
        self.as_of_seq, self.conv_id, self.checker = as_of_seq, conv_id, checker

    def search_transcript(
        self, query: str, k: int = 8, speaker: str | None = None
    ) -> list[ScoredTurn]:
        return self.retriever.search(
            self.conv_id, query, self.as_of_seq, k=k, speaker=speaker
        )

    def get_turns(self, turn_ids: list[str]) -> list[Turn]:
        known = {t.id: t for t in self.store.turns(self.conv_id, self.as_of_seq)}
        wanted = {t.id for i in turn_ids if (t := resolve_turn(known, i)) is not None}
        return [t for t in known.values() if t.id in wanted]

    def lookup_memory(
        self, subject: str, include_superseded: bool = False
    ) -> list[MemoryItem]:
        return self.memory.lookup(
            self.conv_id, subject, self.as_of_seq, include_superseded
        )

    def check_consistency(
        self, statement_turn_id: str, earlier_turn_ids: list[str]
    ) -> tuple[ConsistencyLabel, float]:
        if self.checker is None:
            raise RuntimeError("no consistency checker configured")
        statement = self.get_turns([statement_turn_id])
        if not statement:
            return "insufficient", 0.0
        earlier = [
            t for t in self.get_turns(earlier_turn_ids) if t.seq < statement[0].seq
        ]
        return self.checker.check(statement[0], earlier)
