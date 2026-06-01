from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_evaluating_rag.domain.entities.document import Document
from self_evaluating_rag.domain.exceptions.domain_errors import DocumentNotFoundError
from self_evaluating_rag.domain.ports.document_repository import DocumentRepository


@dataclass(frozen=True)
class DocumentSummary:
    document_id: UUID
    filename: str
    uploaded_at: datetime
    chunk_count: int


@dataclass(frozen=True)
class DocumentDetail:
    document_id: UUID
    filename: str
    uploaded_at: datetime
    chunk_count: int
    content_hash: str
    chroma_collection: str


class ListDocumentsUseCase:
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    async def execute(self, limit: int = 100, offset: int = 0) -> list[DocumentSummary]:
        documents = await self._document_repository.list_all(limit=limit, offset=offset)
        return [_to_summary(document) for document in documents]


class GetDocumentUseCase:
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    async def execute(self, document_id: UUID) -> DocumentDetail:
        document = await self._document_repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(str(document_id))
        return _to_detail(document)


def _to_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.id,
        filename=document.filename,
        uploaded_at=document.created_at,
        chunk_count=document.chunk_count,
    )


def _to_detail(document: Document) -> DocumentDetail:
    return DocumentDetail(
        document_id=document.id,
        filename=document.filename,
        uploaded_at=document.created_at,
        chunk_count=document.chunk_count,
        content_hash=document.content_hash,
        chroma_collection=document.chroma_collection,
    )
