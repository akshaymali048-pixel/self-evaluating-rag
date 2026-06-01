from abc import ABC, abstractmethod
from uuid import UUID

from self_evaluating_rag.domain.entities.document import DocumentChunk
from self_evaluating_rag.domain.entities.query import RetrievedContext


class VectorStore(ABC):
    @abstractmethod
    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: UUID | None = None,
    ) -> list[RetrievedContext]:
        pass
