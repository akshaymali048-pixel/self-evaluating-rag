import asyncio
from uuid import UUID

from self_evaluating_rag.domain.entities.document import DocumentChunk
from self_evaluating_rag.domain.entities.query import RetrievedContext
from self_evaluating_rag.domain.ports.vector_store import VectorStore
from self_evaluating_rag.infrastructure.persistence.chroma.chroma_client_factory import (
    ChromaClientFactory,
)


class ChromaVectorStore(VectorStore):
    def __init__(self, factory: ChromaClientFactory) -> None:
        self._collection = factory.get_or_create_collection()

    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        await asyncio.to_thread(self._add_chunks_sync, chunks, embeddings)

    def _add_chunks_sync(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "page": chunk.page if chunk.page is not None else -1,
            }
            for chunk in chunks
        ]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: UUID | None = None,
    ) -> list[RetrievedContext]:
        return await asyncio.to_thread(
            self._similarity_search_sync,
            query_embedding,
            top_k,
            document_id,
        )

    def _similarity_search_sync(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: UUID | None,
    ) -> list[RetrievedContext]:
        where_filter = None
        if document_id is not None:
            where_filter = {"document_id": str(document_id)}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        contexts: list[RetrievedContext] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            distance = distances[idx] if idx < len(distances) else 1.0
            score = 1.0 - min(max(distance, 0.0), 1.0)
            doc_id_str = metadata.get("document_id")
            parsed_doc_id = UUID(doc_id_str) if doc_id_str else None
            contexts.append(
                RetrievedContext(
                    chunk_id=chunk_id,
                    text=documents[idx] or "",
                    score=score,
                    document_id=parsed_doc_id,
                    metadata=metadata,
                )
            )
        return contexts
