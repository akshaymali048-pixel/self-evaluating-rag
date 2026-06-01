from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievedContext:
    chunk_id: str
    text: str
    score: float
    document_id: UUID | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class Question:
    text: str
    document_id: UUID | None = None
    top_k: int | None = None
    ground_truth: str | None = None
