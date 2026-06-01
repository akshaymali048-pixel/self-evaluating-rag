from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    chunk_count: int
    content_hash: str
    message: str


class DocumentSummaryResponse(BaseModel):
    document_id: UUID
    filename: str
    uploaded_at: datetime = Field(..., description="UTC upload timestamp.")
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]
    total: int = Field(..., description="Number of documents returned in this page.")


class DocumentDetailResponse(BaseModel):
    document_id: UUID
    filename: str
    uploaded_at: datetime
    chunk_count: int
    content_hash: str
    chroma_collection: str
