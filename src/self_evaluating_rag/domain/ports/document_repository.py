from abc import ABC, abstractmethod
from uuid import UUID

from self_evaluating_rag.domain.entities.document import Document


class DocumentRepository(ABC):
    @abstractmethod
    async def save(self, document: Document) -> Document:
        pass

    @abstractmethod
    async def get_by_id(self, document_id: UUID) -> Document | None:
        pass

    @abstractmethod
    async def exists_by_hash(self, content_hash: str) -> bool:
        pass

    @abstractmethod
    async def get_by_hash(self, content_hash: str) -> Document | None:
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        pass

    @abstractmethod
    async def get_latest(self) -> Document | None:
        pass

    @abstractmethod
    async def get_by_filename(self, filename: str) -> Document | None:
        pass
