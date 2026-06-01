from dataclasses import dataclass, field
from uuid import UUID

from self_evaluating_rag.domain.entities.query import RetrievedContext


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    contexts: list[RetrievedContext]
    document_id: UUID | None = None
    model_name: str = ""
