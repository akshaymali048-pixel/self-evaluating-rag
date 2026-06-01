from datetime import datetime
from uuid import UUID

from self_evaluating_rag.domain.ports.evaluation_repository import EvaluationRepository, MetricsSummary


class GetDashboardMetricsUseCase:
    def __init__(self, evaluation_repository: EvaluationRepository) -> None:
        self._evaluation_repository = evaluation_repository

    async def execute(
        self,
        since: datetime | None = None,
        document_id: UUID | None = None,
    ) -> MetricsSummary:
        return await self._evaluation_repository.get_metrics_summary(
            since=since,
            document_id=document_id,
        )
