from typing import Protocol

from contextlens.schemas import Turn


class TranscriptStore(Protocol):
    def add(self, turn: Turn) -> None: ...

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]: ...

    def relabel(self, conv_id: str, seq: int, speaker: str, confidence: float) -> None: ...

    def delete(self, conv_id: str) -> None: ...


class InMemoryTranscriptStore:
    def __init__(self) -> None:
        self._by_conv: dict[str, list[Turn]] = {}

    def add(self, turn: Turn) -> None:
        self._by_conv.setdefault(turn.conversation_id, []).append(turn)

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]:
        found = [t for t in self._by_conv.get(conv_id, []) if start_seq <= t.seq <= as_of_seq]
        return sorted(found, key=lambda t: t.seq)

    def relabel(self, conv_id: str, seq: int, speaker: str, confidence: float) -> None:
        turns = self._by_conv.get(conv_id, [])
        for i, t in enumerate(turns):
            if t.seq == seq:
                turns[i] = t.model_copy(update={"speaker": speaker, "speaker_confidence": confidence})

    def delete(self, conv_id: str) -> None:
        self._by_conv.pop(conv_id, None)


class PostgresTranscriptStore:
    """Turns and their embeddings in PostgreSQL with pgvector (see docker-compose.yml)."""

    def __init__(self, dsn: str, embedder, dim: int = 768) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.embedder = embedder
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS turns (
                id text PRIMARY KEY, conversation_id text NOT NULL, seq int NOT NULL,
                speaker text NOT NULL, speaker_confidence double precision NOT NULL,
                text text NOT NULL, start_ms int NOT NULL, end_ms int NOT NULL,
                source text NOT NULL, embedding vector({dim}) NOT NULL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS turns_conv ON turns (conversation_id, seq)")

    def add(self, turn: Turn) -> None:
        vec = self.embedder.embed([turn.text])[0]
        self._conn.execute(
            "INSERT INTO turns VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (turn.id, turn.conversation_id, turn.seq, turn.speaker, turn.speaker_confidence,
             turn.text, turn.start_ms, turn.end_ms, turn.source, vec),
        )  # fmt: skip

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT id, conversation_id, seq, speaker, speaker_confidence, text, start_ms, end_ms, source"
            " FROM turns WHERE conversation_id=%s AND seq BETWEEN %s AND %s ORDER BY seq",
            (conv_id, start_seq, as_of_seq),
        ).fetchall()
        keys = list(Turn.model_fields)
        return [Turn(**dict(zip(keys, r, strict=True))) for r in rows]

    def vectors(self, conv_id: str, ids: list[str]) -> dict[str, list[float]]:
        rows = self._conn.execute(
            "SELECT id, embedding FROM turns WHERE conversation_id=%s AND id = ANY(%s)", (conv_id, ids)
        ).fetchall()
        return {i: [float(x) for x in v.to_list()] for i, v in rows}

    def relabel(self, conv_id: str, seq: int, speaker: str, confidence: float) -> None:
        self._conn.execute(
            "UPDATE turns SET speaker=%s, speaker_confidence=%s WHERE conversation_id=%s AND seq=%s",
            (speaker, confidence, conv_id, seq),
        )

    def delete(self, conv_id: str) -> None:
        self._conn.execute("DELETE FROM turns WHERE conversation_id=%s", (conv_id,))
