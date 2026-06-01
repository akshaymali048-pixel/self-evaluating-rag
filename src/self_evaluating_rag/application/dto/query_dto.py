from dataclasses import dataclass
from uuid import UUID

from self_evaluating_rag.application.services.evaluation_mode import (
    EvaluationMode,
    normalize_ground_truth,
    resolve_evaluation_mode,
)
from self_evaluating_rag.domain.value_objects.metric_scores import RagasScores


@dataclass(frozen=True)
class AskQuestionInput:
    question: str
    document_id: UUID | None = None
    filename: str | None = None
    use_latest_document: bool = False
    top_k: int | None = None
    ground_truth: str | None = None


@dataclass(frozen=True)
class AskQuestionOutput:
    evaluation_id: UUID
    answer: str
    contexts: list[str]
    scores: RagasScores
    is_hallucination: bool
    latency_ms: int
    model_name: str
    evaluation_mode: EvaluationMode
    document_id: UUID | None = None
    filename: str | None = None
