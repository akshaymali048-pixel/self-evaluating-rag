from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings

from self_evaluating_rag.config.settings import Settings


class ChromaClientFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._persist_dir = Path(settings.chroma_persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
