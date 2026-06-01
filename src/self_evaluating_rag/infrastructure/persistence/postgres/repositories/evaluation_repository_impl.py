from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.domain.entities.evaluation import EvaluationResult
from self_evaluating_rag.domain.ports.evaluation_repository import EvaluationRepository, MetricsSummary
from self_evaluating_rag.infrastructure.persistence.postgres.mappers.evaluation_mapper import (
    EvaluationMapper,
)
from self_evaluating_rag.infrastructure.persistence.postgres.models.evaluation_orm import (
    EvaluationRunORM,
)


class EvaluationRepositoryImpl(EvaluationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, evaluation: EvaluationResult) -> EvaluationResult:
        orm = EvaluationMapper.to_orm(evaluation)
        self._session.add(orm)
        await self._session.flush()
        return evaluation

    async def get_metrics_summary(
        self,
        since: datetime | None = None,
        document_id: UUID | None = None,
    ) -> MetricsSummary:
        filters = []
        if since is not None:
            filters.append(EvaluationRunORM.created_at >= since)
        if document_id is not None:
            filters.append(EvaluationRunORM.document_id == document_id)

        def _apply(stmt):
            return stmt.where(*filters) if filters else stmt

        agg_stmt = _apply(
            select(
                func.count(EvaluationRunORM.id),
                func.avg(EvaluationRunORM.faithfulness),
                func.avg(EvaluationRunORM.answer_relevancy),
                func.avg(EvaluationRunORM.context_precision),
                func.avg(EvaluationRunORM.context_recall),
                func.avg(
                    case((EvaluationRunORM.is_hallucination.is_(True), 1.0), else_=0.0)
                ),
                func.avg(EvaluationRunORM.latency_ms),
            )
        )
        result = await self._session.execute(agg_stmt)
        row = result.one()

        total = int(row[0] or 0)
        hallucination_rate = float(row[5] or 0.0) if total > 0 else 0.0

        p95_stmt = _apply(
            select(
                func.percentile_cont(0.95).within_group(EvaluationRunORM.latency_ms)
            )
        )
        p95_result = await self._session.execute(p95_stmt)
        p95 = p95_result.scalar_one_or_none()

        return MetricsSummary(
            total_runs=total,
            average_faithfulness=float(row[1]) if row[1] is not None else None,
            average_answer_relevancy=float(row[2]) if row[2] is not None else None,
            average_context_precision=float(row[3]) if row[3] is not None else None,
            average_context_recall=float(row[4]) if row[4] is not None else None,
            hallucination_rate=hallucination_rate,
            average_latency_ms=float(row[6] or 0.0),
            p95_latency_ms=float(p95) if p95 is not None else None,
        )
