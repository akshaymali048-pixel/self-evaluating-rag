from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IngestDocumentInput:
    filename: str
    file_path: str


@dataclass(frozen=True)
class IngestDocumentOutput:
    document_id: UUID
    filename: str
    chunk_count: int
    content_hash: str
    message: str
