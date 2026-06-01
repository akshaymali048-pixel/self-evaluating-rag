from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.domain.entities.document import Document
from self_evaluating_rag.domain.ports.document_repository import DocumentRepository
from self_evaluating_rag.infrastructure.persistence.postgres.mappers.document_mapper import (
    DocumentMapper,
)
from self_evaluating_rag.infrastructure.persistence.postgres.models.document_orm import DocumentORM


class DocumentRepositoryImpl(DocumentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, document: Document) -> Document:
        orm = DocumentMapper.to_orm(document)
        self._session.add(orm)
        await self._session.flush()
        return document

    async def get_by_id(self, document_id: UUID) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        orm = result.scalar_one_or_none()
        return DocumentMapper.to_domain(orm) if orm else None

    async def exists_by_hash(self, content_hash: str) -> bool:
        return await self.get_by_hash(content_hash) is not None

    async def get_by_hash(self, content_hash: str) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.content_hash == content_hash).limit(1)
        )
        orm = result.scalar_one_or_none()
        return DocumentMapper.to_domain(orm) if orm else None

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        result = await self._session.execute(
            select(DocumentORM)
            .order_by(DocumentORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [DocumentMapper.to_domain(orm) for orm in result.scalars().all()]

    async def get_latest(self) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM).order_by(DocumentORM.created_at.desc()).limit(1)
        )
        orm = result.scalar_one_or_none()
        return DocumentMapper.to_domain(orm) if orm else None

    async def get_by_filename(self, filename: str) -> Document | None:
        result = await self._session.execute(
            select(DocumentORM)
            .where(DocumentORM.filename == filename)
            .order_by(DocumentORM.created_at.desc())
            .limit(1)
        )
        orm = result.scalar_one_or_none()
        return DocumentMapper.to_domain(orm) if orm else None
