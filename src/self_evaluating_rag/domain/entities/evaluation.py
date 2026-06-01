from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_evaluating_rag.domain.value_objects.metric_scores import RagasScores


@dataclass(frozen=True)
class EvaluationResult:
    id: UUID
    question: str
    answer: str
    contexts: list[str]
    scores: RagasScores
    is_hallucination: bool
    latency_ms: int
    model_name: str
    embedding_model: str
    document_id: UUID | None = None
    ground_truth: str | None = None
    created_at: datetime | None = None
