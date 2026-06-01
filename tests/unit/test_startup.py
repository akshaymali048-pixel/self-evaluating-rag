"""Unit tests that do not require Gemini or Postgres."""

import pytest


def test_settings_optional_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from self_evaluating_rag.config.settings import Settings

    settings = Settings()
    assert settings.llm_provider == "openai"
    assert settings.gemini_configured is False
    assert settings.openai_configured is False
    assert settings.ai_provider_configured is False


def test_openai_required_on_embed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from self_evaluating_rag.config.settings import Settings
    from self_evaluating_rag.domain.exceptions.domain_errors import OpenAINotConfiguredError
    from self_evaluating_rag.infrastructure.llm.openai_embeddings import OpenAIEmbeddingService

    service = OpenAIEmbeddingService(Settings())
    with pytest.raises(OpenAINotConfiguredError):
        service._get_embeddings()


def test_gemini_required_on_embed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from self_evaluating_rag.config.settings import Settings
    from self_evaluating_rag.domain.exceptions.domain_errors import GeminiNotConfiguredError
    from self_evaluating_rag.infrastructure.llm.gemini_embeddings import GeminiEmbeddingService

    service = GeminiEmbeddingService(Settings())
    with pytest.raises(GeminiNotConfiguredError):
        service._get_embeddings()
