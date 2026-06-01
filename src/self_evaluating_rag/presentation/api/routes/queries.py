from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.application.dto.query_dto import AskQuestionOutput
from self_evaluating_rag.domain.exceptions.domain_errors import (
    DocumentNotFoundError,
    EvaluationError,
    GeminiNotConfiguredError,
    GeminiQuotaExceededError,
    NoDocumentsError,
    OpenAINotConfiguredError,
    RetrievalError,
)
from self_evaluating_rag.presentation.api.dependencies import get_container, get_session
from self_evaluating_rag.presentation.api.schemas.query_schemas import (
    AskByFilenameQuestionRequest,
    AskLatestQuestionRequest,
    AskQuestionRequest,
    AskQuestionResponse,
    RagasScoresResponse,
)

router = APIRouter(prefix="/api/v1/queries", tags=["queries"])


def _to_ask_response(result: AskQuestionOutput) -> AskQuestionResponse:
    return AskQuestionResponse(
        evaluation_id=result.evaluation_id,
        answer=result.answer,
        contexts=result.contexts,
        scores=RagasScoresResponse(
            faithfulness=result.scores.faithfulness,
            answer_relevancy=result.scores.answer_relevancy,
            context_precision=result.scores.context_precision,
            context_recall=result.scores.context_recall,
        ),
        is_hallucination=result.is_hallucination,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
        evaluation_mode=result.evaluation_mode.value,
        document_id=result.document_id,
        filename=result.filename,
    )


async def _run_ask(container, session, ask_input):
    use_case = container.ask_question_use_case(session)
    try:
        return await use_case.execute(ask_input)
    except (GeminiNotConfiguredError, OpenAINotConfiguredError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except GeminiQuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except NoDocumentsError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EvaluationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post(
    "/ask",
    response_model=AskQuestionResponse,
    summary="Ask by document UUID",
    description=(
        "Backward-compatible endpoint. Provide `document_id` explicitly, or omit it to "
        "search across all indexed chunks. Supports User Mode and Evaluation Mode."
    ),
)
async def ask_question(
    request: Request,
    body: AskQuestionRequest,
    session: AsyncSession = Depends(get_session),
) -> AskQuestionResponse:
    from self_evaluating_rag.application.dto.query_dto import AskQuestionInput

    container = get_container(request)
    result = await _run_ask(
        container,
        session,
        AskQuestionInput(
            question=body.question,
            document_id=body.document_id,
            top_k=body.top_k,
            ground_truth=body.ground_truth,
        ),
    )
    return _to_ask_response(result)


@router.post(
    "/ask-latest",
    response_model=AskQuestionResponse,
    summary="Ask against the latest uploaded document",
    description=(
        "Automatically resolves the most recently uploaded PDF. "
        "No document UUID required. Supports User Mode and Evaluation Mode."
    ),
)
async def ask_latest_question(
    request: Request,
    body: AskLatestQuestionRequest,
    session: AsyncSession = Depends(get_session),
) -> AskQuestionResponse:
    from self_evaluating_rag.application.dto.query_dto import AskQuestionInput

    container = get_container(request)
    result = await _run_ask(
        container,
        session,
        AskQuestionInput(
            question=body.question,
            use_latest_document=True,
            top_k=body.top_k,
            ground_truth=body.ground_truth,
        ),
    )
    return _to_ask_response(result)


@router.post(
    "/ask-by-filename",
    response_model=AskQuestionResponse,
    summary="Ask by original filename",
    description=(
        "Resolves a document by its upload filename (e.g. Resume.pdf). "
        "When multiple uploads share a filename, the most recent is used."
    ),
)
async def ask_by_filename_question(
    request: Request,
    body: AskByFilenameQuestionRequest,
    session: AsyncSession = Depends(get_session),
) -> AskQuestionResponse:
    from self_evaluating_rag.application.dto.query_dto import AskQuestionInput

    container = get_container(request)
    result = await _run_ask(
        container,
        session,
        AskQuestionInput(
            question=body.question,
            filename=body.filename,
            top_k=body.top_k,
            ground_truth=body.ground_truth,
        ),
    )
    return _to_ask_response(result)
