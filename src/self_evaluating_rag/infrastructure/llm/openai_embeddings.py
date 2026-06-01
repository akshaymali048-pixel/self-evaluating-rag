import asyncio

from langchain_openai import OpenAIEmbeddings

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.ports.embedding_service import EmbeddingService
from self_evaluating_rag.infrastructure.llm.openai_client import create_embedding_model


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings: OpenAIEmbeddings | None = None

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = create_embedding_model(self._settings)
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self._settings.openai_embedding_model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._get_embeddings().embed_documents, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_embeddings().embed_query, text)
