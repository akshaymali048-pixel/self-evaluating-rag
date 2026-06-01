import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.application.dto.ingest_dto import IngestDocumentInput
from self_evaluating_rag.config.settings import Settings, get_settings
from self_evaluating_rag.domain.exceptions.domain_errors import (
    DocumentNotFoundError,
    GeminiNotConfiguredError,
    IngestionError,
    OpenAINotConfiguredError,
)
from self_evaluating_rag.presentation.api.dependencies import get_container, get_session
from self_evaluating_rag.presentation.api.schemas.document_schemas import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DocumentListResponse:
    container = get_container(request)
    use_case = container.list_documents_use_case(session)
    documents = await use_case.execute(limit=limit, offset=offset)
    summaries = [
        DocumentSummaryResponse(
            document_id=document.document_id,
            filename=document.filename,
            uploaded_at=document.uploaded_at,
            chunk_count=document.chunk_count,
        )
        for document in documents
    ]
    return DocumentListResponse(documents=summaries, total=len(summaries))


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    request: Request,
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentDetailResponse:
    container = get_container(request)
    use_case = container.get_document_use_case(session)
    try:
        document = await use_case.execute(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return DocumentDetailResponse(
        document_id=document.document_id,
        filename=document.filename,
        uploaded_at=document.uploaded_at,
        chunk_count=document.chunk_count,
        content_hash=document.content_hash,
        chroma_collection=document.chroma_collection,
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    stored_path = upload_dir / f"{uuid.uuid4()}_{safe_name}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    stored_path.write_bytes(content)

    container = get_container(request)
    use_case = container.ingest_document_use_case(session)

    try:
        result = await use_case.execute(
            IngestDocumentInput(filename=safe_name, file_path=str(stored_path))
        )
    except (GeminiNotConfiguredError, OpenAINotConfiguredError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        content_hash=result.content_hash,
        message=result.message,
    )
