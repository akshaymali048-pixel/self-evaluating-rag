from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
