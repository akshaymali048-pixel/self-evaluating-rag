"""Lazy OpenAI client factory — no API calls at import time."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.exceptions.domain_errors import OpenAINotConfiguredError


def require_openai_api_key(settings: Settings) -> str:
    if not settings.openai_configured:
        raise OpenAINotConfiguredError(
            "OPENAI_API_KEY is not set. Configure it when LLM_PROVIDER=openai."
        )
    return settings.openai_api_key.strip()  # type: ignore[union-attr]


def create_chat_model(settings: Settings, *, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=require_openai_api_key(settings),
        temperature=temperature,
    )


def create_embedding_model(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=require_openai_api_key(settings),
    )
