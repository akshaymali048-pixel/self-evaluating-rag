from self_evaluating_rag.domain.entities.document import Document
from self_evaluating_rag.infrastructure.persistence.postgres.models.document_orm import DocumentORM


class DocumentMapper:
    @staticmethod
    def to_domain(orm: DocumentORM) -> Document:
        return Document(
            id=orm.id,
            filename=orm.filename,
            content_hash=orm.content_hash,
            chunk_count=orm.chunk_count,
            chroma_collection=orm.chroma_collection,
            created_at=orm.created_at,
        )

    @staticmethod
    def to_orm(entity: Document) -> DocumentORM:
        return DocumentORM(
            id=entity.id,
            filename=entity.filename,
            content_hash=entity.content_hash,
            chunk_count=entity.chunk_count,
            chroma_collection=entity.chroma_collection,
            created_at=entity.created_at,
        )
