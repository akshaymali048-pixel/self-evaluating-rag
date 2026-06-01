from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_evaluating_rag.domain.entities.evaluation import EvaluationResult


@dataclass(frozen=True)
class MetricsSummary:
    total_runs: int
    average_faithfulness: float | None
    average_answer_relevancy: float | None
    average_context_precision: float | None
    average_context_recall: float | None
    hallucination_rate: float
    average_latency_ms: float
    p95_latency_ms: float | None


class EvaluationRepository(ABC):
    @abstractmethod
    async def save(self, evaluation: EvaluationResult) -> EvaluationResult:
        pass

    @abstractmethod
    async def get_metrics_summary(
        self,
        since: datetime | None = None,
        document_id: UUID | None = None,
    ) -> MetricsSummary:
        pass
