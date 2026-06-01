# Dependency pinning rationale

This project pins versions in `requirements.txt` for reproducible installs aligned with **RAGAS 0.4.3**.

## RAGAS

| Package | Version | Why |
|---------|---------|-----|
| **ragas** | `0.4.3` | Target evaluation API (`EvaluationDataset`, `SingleTurnSample`, `metrics.collections`). Newer 0.4.x is compatible; this is the latest 0.4 release tested. |
| **datasets** | `4.0.0` | RAGAS 0.4.3 declares `datasets>=4.0.0`. v4 matches the `EvaluationDataset` / HF dataset bridge used in evaluation. |

## LangChain

| Package | Version | Why |
|---------|---------|-----|
| **langchain** | `0.3.25` | RAGAS 0.4 depends on LangChain but does not cap the major version. The **0.3** line avoids breaking changes in LangChain 1.x while satisfying RAGAS imports. |
| **langchain-core** | `0.3.65` | Message types (`HumanMessage`, `SystemMessage`) and Runnable protocol used by `langchain-google-genai`; kept on the same 0.3 minor family as `langchain`. |
| **langchain-community** | `0.3.25` | Required by RAGAS 0.4; version-aligned with `langchain` 0.3.25 to reduce resolver conflicts. |
| **langchain-google-genai** | `2.1.4` | Gemini chat + embeddings for RAG and RAGAS wrappers; 2.x works with `langchain-core` 0.3 (4.x targets LangChain 1.x). |
| **langchain-text-splitters** | `0.3.8` | `RecursiveCharacterTextSplitter` for PDF chunking; matched to LangChain 0.3 ecosystem. |

## ChromaDB

| Package | Version | Why |
|---------|---------|-----|
| **chromadb** | `0.6.3` | Code uses `PersistentClient`, `get_or_create_collection`, and manual embeddings. **0.6.x** is stable for this pattern; **1.x** introduced client/API shifts that the current adapter was not written for. |

## API & SQL

| Package | Version | Why |
|---------|---------|-----|
| **fastapi** | `0.115.12` | Mature 0.115 line with Pydantic v2 and lifespan support used in `main.py`. |
| **sqlalchemy** | `2.0.41` | Async engine + `mapped_column` ORM models and Alembic migrations. |
| **asyncpg** | `0.30.0` | PostgreSQL async driver for `postgresql+asyncpg://` URLs. |
| **alembic** | `1.16.2` | Async migration runner in `alembic/env.py`. |

## Supporting packages

| Package | Version | Why |
|---------|---------|-----|
| **uvicorn**, **python-multipart**, **pydantic-settings** | pinned | ASGI server, file upload, and `.env` loading. |
| **pypdf** | `5.6.0` | PDF text extraction in ingestion. |
| **httpx** | `0.28.1` | HTTP client used by LangChain/RAGAS stacks. |
| **pytest**, **pytest-asyncio** | pinned | Integration tests for the full pipeline. |

## What we intentionally avoid

- **LangChain 1.x + langchain-google-genai 4.x** — resolver may pick them with open `>=` pins; they change import paths and RAGAS wrapper behavior.
- **ChromaDB 1.x** — until the vector store adapter is updated and tested.
- **datasets 3.x** — incompatible with RAGAS 0.4.3 lower bound.
