from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.presentation.api.dependencies import get_container, get_session
from self_evaluating_rag.presentation.api.schemas.evaluation_schemas import MetricsSummaryResponse

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("/summary", response_model=MetricsSummaryResponse)
async def metrics_summary(
    request: Request,
    session: AsyncSession = Depends(get_session),
    since: datetime | None = Query(None, description="Filter evaluations created after this time."),
    document_id: UUID | None = Query(None, description="Filter by document ID."),
) -> MetricsSummaryResponse:
    container = get_container(request)
    use_case = container.get_dashboard_metrics_use_case(session)
    summary = await use_case.execute(since=since, document_id=document_id)

    return MetricsSummaryResponse(
        total_runs=summary.total_runs,
        average_faithfulness=summary.average_faithfulness,
        average_answer_relevancy=summary.average_answer_relevancy,
        average_context_precision=summary.average_context_precision,
        average_context_recall=summary.average_context_recall,
        hallucination_rate=summary.hallucination_rate,
        average_latency_ms=summary.average_latency_ms,
        p95_latency_ms=summary.p95_latency_ms,
    )
