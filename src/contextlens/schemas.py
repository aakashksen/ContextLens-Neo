from typing import Literal, Self

from pydantic import BaseModel, model_validator

ConsistencyLabel = Literal["consistent", "update", "contradiction", "insufficient"]


class Turn(BaseModel):
    id: str
    conversation_id: str
    seq: int
    speaker: str
    speaker_confidence: float = 1.0
    text: str
    start_ms: int
    end_ms: int
    source: Literal["dataset", "synthetic", "asr"]


class Conversation(BaseModel):
    id: str
    turns: list[Turn]
    source: str
    split: Literal["train", "dev", "test"] | None = None


class MemoryItem(BaseModel):
    id: str
    conversation_id: str
    kind: Literal["decision", "commitment", "question", "correction", "fact"]
    subject: str
    statement: str
    speaker: str
    source_turn_ids: list[str]
    supersedes: str | None = None
    as_of_seq: int


class EvidenceRef(BaseModel):
    turn_id: str
    quote: str
    speaker: str
    start_ms: int


class Finding(BaseModel):
    agent: str
    kind: Literal["recall", "consistency", "context"]
    statement: str
    evidence: list[EvidenceRef]
    label: ConsistencyLabel | None = None
    confidence: float

    @model_validator(mode="after")
    def _recall_needs_evidence(self) -> Self:
        if self.kind == "recall" and not self.evidence:
            raise ValueError("a recall finding needs at least one evidence reference")
        return self


class Card(BaseModel):
    mode: Literal["recall", "clarify", "respond"]
    status: Literal["answered", "partial", "abstained"]
    findings: list[Finding]
    suggestion: str | None = None
    missing: list[str] = []
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    @model_validator(mode="after")
    def _answered_needs_finding(self) -> Self:
        if self.status == "answered" and not self.findings:
            raise ValueError("an answered card needs at least one finding")
        return self


class ButtonRequest(BaseModel):
    conversation_id: str
    as_of_seq: int
    question: str | None = None
    detail: Literal["brief", "standard", "detailed"] = "standard"


class Scenario(BaseModel):
    id: str
    conversation: Conversation
    request: ButtonRequest
    category: str
    gold_evidence_turn_ids: list[str]
    gold_label: ConsistencyLabel | None
    gold_answer: str | None
    origin: Literal["dataset", "synthetic"]
