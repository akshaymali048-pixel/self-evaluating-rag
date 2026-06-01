from abc import ABC, abstractmethod

from self_evaluating_rag.domain.entities.document import DocumentChunk


class ChunkingService(ABC):
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        document_id,
        pages: list[tuple[int, str]] | None = None,
    ) -> list[DocumentChunk]:
        pass
