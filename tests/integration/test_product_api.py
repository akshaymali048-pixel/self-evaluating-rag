"""Integration tests for production document discovery and evaluation modes."""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _fresh_client() -> TestClient:
    from self_evaluating_rag.config.settings import get_settings
    from self_evaluating_rag.presentation.api.main import create_app

    get_settings.cache_clear()
    return TestClient(create_app())


def _upload(client: TestClient, pdf_bytes: bytes, filename: str = "pipeline.pdf"):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.integration
def test_list_documents_after_upload(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        uploaded = _upload(client, sample_pdf_bytes, "listed.pdf")
        listing = client.get("/api/v1/documents")
        assert listing.status_code == 200
        body = listing.json()
        assert body["total"] >= 1
        match = next(
            item for item in body["documents"] if item["document_id"] == uploaded["document_id"]
        )
        assert match["filename"] == "listed.pdf"
        assert match["chunk_count"] == uploaded["chunk_count"]
        assert match["uploaded_at"]


@pytest.mark.integration
def test_get_document_by_id(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        uploaded = _upload(client, sample_pdf_bytes, "detail.pdf")
        detail = client.get(f"/api/v1/documents/{uploaded['document_id']}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["document_id"] == uploaded["document_id"]
        assert body["filename"] == "detail.pdf"
        assert body["content_hash"] == uploaded["content_hash"]


def test_get_document_not_found():
    with _fresh_client() as client:
        response = client.get(
            "/api/v1/documents/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404


def test_ask_latest_without_documents(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with _fresh_client() as client:
        response = client.post(
            "/api/v1/queries/ask-latest",
            json={"question": "What is this about?", "top_k": 3},
        )
        assert response.status_code in {404, 503}


@pytest.mark.integration
def test_ask_latest_user_mode(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        _upload(client, sample_pdf_bytes, "latest.pdf")
        response = client.post(
            "/api/v1/queries/ask-latest",
            json={"question": "What is this document about?", "top_k": 3},
            timeout=300,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["evaluation_mode"] == "user"
        assert body["document_id"]
        assert body["filename"] == "latest.pdf"
        assert body["scores"]["faithfulness"] is not None
        assert body["scores"]["answer_relevancy"] is not None
        assert body["scores"]["context_precision"] is None
        assert body["scores"]["context_recall"] is None


@pytest.mark.integration
def test_ask_by_filename_user_mode(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        _upload(client, sample_pdf_bytes, "Resume.pdf")
        response = client.post(
            "/api/v1/queries/ask-by-filename",
            json={
                "filename": "Resume.pdf",
                "question": "What is this document about?",
                "top_k": 3,
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["filename"] == "Resume.pdf"
        assert body["evaluation_mode"] == "user"
        assert body["scores"]["context_precision"] is None


@pytest.mark.integration
def test_ask_with_ground_truth_evaluation_mode(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        uploaded = _upload(client, sample_pdf_bytes, "eval.pdf")
        response = client.post(
            "/api/v1/queries/ask",
            json={
                "question": "What is this document about?",
                "document_id": uploaded["document_id"],
                "top_k": 3,
                "ground_truth": "RAG integration test document.",
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["evaluation_mode"] == "evaluation"
        assert body["scores"]["faithfulness"] is not None
        assert body["scores"]["context_precision"] is not None
        assert body["scores"]["context_recall"] is not None


@pytest.mark.integration
def test_ask_with_empty_ground_truth_user_mode(sample_pdf_bytes, integration_enabled):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    with _fresh_client() as client:
        uploaded = _upload(client, sample_pdf_bytes, "empty-gt.pdf")
        response = client.post(
            "/api/v1/queries/ask",
            json={
                "question": "What is this document about?",
                "document_id": uploaded["document_id"],
                "top_k": 3,
                "ground_truth": "   ",
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["evaluation_mode"] == "user"
        assert body["scores"]["context_precision"] is None
        assert body["scores"]["context_recall"] is None


@pytest.mark.integration
def test_reference_metrics_disabled(sample_pdf_bytes, integration_enabled, monkeypatch):
    if not integration_enabled:
        pytest.skip("Integration tests require API key and RUN_INTEGRATION_TESTS=1")

    monkeypatch.setenv("ENABLE_REFERENCE_METRICS", "false")
    with _fresh_client() as client:
        uploaded = _upload(client, sample_pdf_bytes, "disabled-ref.pdf")
        response = client.post(
            "/api/v1/queries/ask",
            json={
                "question": "What is this document about?",
                "document_id": uploaded["document_id"],
                "top_k": 3,
                "ground_truth": "RAG integration test document.",
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["evaluation_mode"] == "user"
        assert body["scores"]["context_precision"] is None
        assert body["scores"]["context_recall"] is None
