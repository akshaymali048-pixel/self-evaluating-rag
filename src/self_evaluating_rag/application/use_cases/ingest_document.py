import hashlib
import logging
from datetime import UTC, datetime
from uuid import uuid4

from self_evaluating_rag.application.dto.ingest_dto import IngestDocumentInput, IngestDocumentOutput
from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.entities.document import Document
from self_evaluating_rag.domain.exceptions.domain_errors import IngestionError
from self_evaluating_rag.domain.ports.chunking_service import ChunkingService
from self_evaluating_rag.domain.ports.document_repository import DocumentRepository
from self_evaluating_rag.domain.ports.embedding_service import EmbeddingService
from self_evaluating_rag.domain.ports.pdf_loader import PDFLoader
from self_evaluating_rag.domain.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestDocumentUseCase:
    def __init__(
        self,
        pdf_loader: PDFLoader,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        document_repository: DocumentRepository,
        settings: Settings,
    ) -> None:
        self._pdf_loader = pdf_loader
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._document_repository = document_repository
        self._settings = settings

    async def execute(self, request: IngestDocumentInput) -> IngestDocumentOutput:
        try:
            loaded = self._pdf_loader.load(request.file_path)
        except Exception as exc:
            raise IngestionError(f"Failed to load PDF: {exc}") from exc

        content_hash = hashlib.sha256(loaded.full_text.encode("utf-8")).hexdigest()
        existing = await self._document_repository.get_by_hash(content_hash)
        if existing is not None:
            return IngestDocumentOutput(
                document_id=existing.id,
                filename=existing.filename,
                chunk_count=existing.chunk_count,
                content_hash=content_hash,
                message="Document already ingested (duplicate content hash).",
            )

        document_id = uuid4()
        chunks = self._chunking_service.chunk_text(
            loaded.full_text,
            document_id=document_id,
            pages=loaded.pages,
        )
        if not chunks:
            raise IngestionError("No text chunks produced from PDF.")

        texts = [chunk.text for chunk in chunks]
        embeddings = await self._embedding_service.embed_documents(texts)
        await self._vector_store.add_chunks(chunks, embeddings)

        document = Document(
            id=document_id,
            filename=request.filename,
            content_hash=content_hash,
            chunk_count=len(chunks),
            chroma_collection=self._settings.chroma_collection_name,
            created_at=datetime.now(UTC),
        )
        await self._document_repository.save(document)

        logger.info("Ingested document %s with %d chunks", document_id, len(chunks))
        return IngestDocumentOutput(
            document_id=document_id,
            filename=request.filename,
            chunk_count=len(chunks),
            content_hash=content_hash,
            message="Document ingested successfully.",
        )
