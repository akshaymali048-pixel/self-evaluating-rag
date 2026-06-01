from typing import Any

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.ports.embedding_service import EmbeddingService
from self_evaluating_rag.domain.ports.llm_service import LLMService
from self_evaluating_rag.infrastructure.llm.gemini_embeddings import GeminiEmbeddingService
from self_evaluating_rag.infrastructure.llm.gemini_llm import GeminiLLMService
from self_evaluating_rag.infrastructure.llm.openai_embeddings import OpenAIEmbeddingService
from self_evaluating_rag.infrastructure.llm.openai_llm import OpenAILLMService


def create_embedding_service(settings: Settings) -> EmbeddingService:
    if settings.llm_provider == "openai":
        return OpenAIEmbeddingService(settings)
    return GeminiEmbeddingService(settings)


def create_llm_service(settings: Settings) -> LLMService:
    if settings.llm_provider == "openai":
        return OpenAILLMService(settings)
    return GeminiLLMService(settings)


def create_langchain_chat_model(settings: Settings, *, temperature: float) -> Any:
    if settings.llm_provider == "openai":
        from self_evaluating_rag.infrastructure.llm.openai_client import create_chat_model

        return create_chat_model(settings, temperature=temperature)
    from self_evaluating_rag.infrastructure.llm.gemini_client import create_chat_model

    return create_chat_model(settings, temperature=temperature)


def create_langchain_embedding_model(settings: Settings) -> Any:
    if settings.llm_provider == "openai":
        from self_evaluating_rag.infrastructure.llm.openai_client import create_embedding_model

        return create_embedding_model(settings)
    from self_evaluating_rag.infrastructure.llm.gemini_client import create_embedding_model

    return create_embedding_model(settings)
