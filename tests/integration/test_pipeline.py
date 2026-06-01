"""End-to-end API pipeline: upload → embed → ask → RAGAS → persist."""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _fresh_client() -> TestClient:
    from self_evaluating_rag.config.settings import get_settings
    from self_evaluating_rag.presentation.api.main import create_app

    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.mark.integration
def test_app_starts_without_ai_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with _fresh_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert body["llm_provider"] == "openai"
        assert body["openai_configured"] is False
        assert body["ai_provider_configured"] is False


def test_upload_without_openai_key_returns_503(monkeypatch, sample_pdf_bytes):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with _fresh_client() as client:
        files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 503
        assert "OPENAI_API_KEY" in response.json()["detail"]


@pytest.mark.integration
def test_full_rag_pipeline(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip(
            "Set OPENAI_API_KEY (or GEMINI_API_KEY with LLM_PROVIDER=gemini) "
            "and RUN_INTEGRATION_TESTS=1 with Postgres migrated"
        )

    with _fresh_client() as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("pipeline.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["document_id"]
        assert upload.json()["chunk_count"] > 0

        ask = client.post(
            "/api/v1/queries/ask",
            json={
                "question": "What is this document about?",
                "document_id": document_id,
                "top_k": 3,
                "ground_truth": "RAG integration test document.",
            },
            timeout=300,
        )
        assert ask.status_code == 200, ask.text
        body = ask.json()
        assert body["answer"]
        assert len(body["contexts"]) > 0
        assert body["evaluation_id"]
        assert body["scores"]["faithfulness"] is not None
        assert body["scores"]["answer_relevancy"] is not None
        assert body["evaluation_mode"] == "evaluation"
        assert body["latency_ms"] > 0

        metrics = client.get("/api/v1/metrics/summary")
        assert metrics.status_code == 200
        summary = metrics.json()
        assert summary["total_runs"] >= 1
        assert summary["average_faithfulness"] is not None
