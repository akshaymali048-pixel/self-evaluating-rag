from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MetricsSummaryResponse(BaseModel):
    total_runs: int
    average_faithfulness: float | None
    average_answer_relevancy: float | None
    average_context_precision: float | None
    average_context_recall: float | None
    hallucination_rate: float
    average_latency_ms: float
    p95_latency_ms: float | None
