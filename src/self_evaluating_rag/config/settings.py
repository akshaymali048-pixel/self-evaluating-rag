from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field("Self Evaluating RAG API", alias="APP_NAME")
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    llm_provider: LLMProvider = Field("openai", alias="LLM_PROVIDER")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "models/gemini-embedding-001",
        alias="GEMINI_EMBEDDING_MODEL",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    database_url: str = Field(
        "postgresql+asyncpg://rag:rag@localhost:5432/rag_eval",
        alias="DATABASE_URL",
    )

    chroma_persist_dir: Path = Field(Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("rag_chunks", alias="CHROMA_COLLECTION_NAME")

    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(5, alias="RETRIEVAL_TOP_K")
    upload_dir: Path = Field(Path("./data/uploads"), alias="UPLOAD_DIR")

    hallucination_faithfulness_threshold: float = Field(
        0.5,
        alias="HALLUCINATION_FAITHFULNESS_THRESHOLD",
    )
    ragas_timeout_seconds: int = Field(120, alias="RAGAS_TIMEOUT_SECONDS")
    enable_reference_metrics: bool = Field(True, alias="ENABLE_REFERENCE_METRICS")

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def ai_provider_configured(self) -> bool:
        if self.llm_provider == "openai":
            return self.openai_configured
        return self.gemini_configured


@lru_cache
def get_settings() -> Settings:
    return Settings()
