from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EvaluationModeName = Literal["user", "evaluation"]

USER_MODE_DESCRIPTION = (
    "User Mode: ask a question without ground_truth. "
    "Runs faithfulness, answer_relevancy, and hallucination detection only."
)
EVALUATION_MODE_DESCRIPTION = (
    "Evaluation Mode: provide ground_truth with ENABLE_REFERENCE_METRICS=true. "
    "Runs full RAGAS metrics including context_precision and context_recall."
)


class AskQuestionRequest(BaseModel):
    """Ask against a specific document by UUID (backward-compatible)."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "summary": "User Mode",
                "description": USER_MODE_DESCRIPTION,
                "value": {
                    "question": "What are the key skills listed?",
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "top_k": 5,
                },
            },
            {
                "summary": "Evaluation Mode",
                "description": EVALUATION_MODE_DESCRIPTION,
                "value": {
                    "question": "What are the key skills listed?",
                    "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "top_k": 5,
                    "ground_truth": "Python, FastAPI, and PostgreSQL.",
                },
            },
        ]
    })

    question: str = Field(..., min_length=1, max_length=4096)
    document_id: UUID | None = Field(
        None,
        description="Target document UUID. Omit only when searching all indexed chunks.",
    )
    top_k: int | None = Field(None, ge=1, le=20)
    ground_truth: str | None = Field(
        None,
        description=(
            "Optional reference answer. When provided and ENABLE_REFERENCE_METRICS=true, "
            "enables Evaluation Mode with context_precision and context_recall."
        ),
    )


class AskLatestQuestionRequest(BaseModel):
    """Ask against the most recently uploaded document (no UUID required)."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "summary": "User Mode",
                "value": {"question": "Summarize this document.", "top_k": 5},
            },
            {
                "summary": "Evaluation Mode",
                "value": {
                    "question": "Summarize this document.",
                    "top_k": 5,
                    "ground_truth": "A one-paragraph summary of the uploaded PDF.",
                },
            },
        ]
    })

    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int | None = Field(None, ge=1, le=20)
    ground_truth: str | None = Field(
        None,
        description="Optional reference answer for Evaluation Mode.",
    )


class AskByFilenameQuestionRequest(BaseModel):
    """Ask against a document resolved by its original filename."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "summary": "User Mode",
                "value": {
                    "filename": "Resume.pdf",
                    "question": "What is the candidate's experience?",
                    "top_k": 5,
                },
            },
            {
                "summary": "Evaluation Mode",
                "value": {
                    "filename": "Resume.pdf",
                    "question": "What is the candidate's experience?",
                    "top_k": 5,
                    "ground_truth": "5 years of backend engineering experience.",
                },
            },
        ]
    })

    filename: str = Field(..., min_length=1, max_length=512)
    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int | None = Field(None, ge=1, le=20)
    ground_truth: str | None = Field(
        None,
        description="Optional reference answer for Evaluation Mode.",
    )


class RagasScoresResponse(BaseModel):
    faithfulness: float | None = Field(
        None, description="Always computed in User Mode and Evaluation Mode."
    )
    answer_relevancy: float | None = Field(
        None, description="Always computed in User Mode and Evaluation Mode."
    )
    context_precision: float | None = Field(
        None,
        description="Evaluation Mode only. Null in User Mode.",
    )
    context_recall: float | None = Field(
        None,
        description="Evaluation Mode only. Null in User Mode.",
    )


class AskQuestionResponse(BaseModel):
    evaluation_id: UUID
    answer: str
    contexts: list[str]
    scores: RagasScoresResponse
    is_hallucination: bool
    latency_ms: int
    model_name: str
    evaluation_mode: EvaluationModeName = Field(
        ...,
        description="`user` when no reference metrics ran; `evaluation` when ground_truth metrics ran.",
    )
    document_id: UUID | None = Field(
        None,
        description="Resolved document UUID used for retrieval.",
    )
    filename: str | None = Field(
        None,
        description="Resolved filename when available.",
    )
