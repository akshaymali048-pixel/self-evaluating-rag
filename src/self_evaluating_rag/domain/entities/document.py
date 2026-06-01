from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: UUID
    text: str
    page: int | None = None
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    id: UUID
    filename: str
    content_hash: str
    chunk_count: int
    chroma_collection: str
    created_at: datetime
