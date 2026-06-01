"""Lazy Gemini client factory — no API calls or client construction at import time."""

from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.exceptions.domain_errors import (
    GeminiNotConfiguredError,
    GeminiQuotaExceededError,
)


def require_gemini_api_key(settings: Settings) -> str:
    if not settings.gemini_configured:
        raise GeminiNotConfiguredError(
            "GEMINI_API_KEY is not set. Configure it to use embeddings, generation, or RAGAS."
        )
    return settings.gemini_api_key.strip()  # type: ignore[union-attr]


def create_chat_model(settings: Settings, *, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=require_gemini_api_key(settings),
        temperature=temperature,
    )


def create_embedding_model(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=require_gemini_api_key(settings),
    )


def raise_if_gemini_quota_exceeded(exc: BaseException, *, model: str | None = None) -> None:
    """Map Google ResourceExhausted (429) to a domain error, including wrapped causes."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ResourceExhausted):
            raise GeminiQuotaExceededError(model=model) from exc
        current = current.__cause__ or current.__context__
