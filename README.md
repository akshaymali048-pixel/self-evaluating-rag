# Self-Evaluating RAG API

Production-oriented FastAPI backend for PDF ingestion, ChromaDB retrieval, Gemini generation, and RAGAS self-evaluation with PostgreSQL persistence.

## Quick start

```bash
# Full stack (Postgres + API with migrations on startup)
cp .env.example .env
# Optional: set GEMINI_API_KEY in .env for ingest/query
docker compose up --build

# Local development
pip install -r requirements.txt
pip install -e .
alembic upgrade head
uvicorn self_evaluating_rag.presentation.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API starts without `GEMINI_API_KEY`; upload and query return **503** until the key is set.

### Integration tests

```bash
docker compose up -d postgres
alembic upgrade head
export GEMINI_API_KEY=your_key
export RUN_INTEGRATION_TESTS=1
pytest tests/ -v
```

See `DEPENDENCIES.md` for pinned version rationale.

API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/documents/upload` | Upload PDF, chunk, embed, store |
| POST | `/api/v1/queries/ask` | RAG query + RAGAS evaluation |
| GET | `/api/v1/metrics/summary` | Aggregated evaluation metrics |

## Architecture

PDF Upload
    ↓
Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Store
    ↓
Retriever
    ↓
Gemini LLM
    ↓
RAGAS Evaluation
    ↓
Metrics Persistence (PostgreSQL)

## Features

- PDF ingestion
- Vector search using ChromaDB
- Gemini-powered answer generation
- Faithfulness evaluation
- Answer relevancy evaluation
- Hallucination detection
- Context precision evaluation
- Context recall evaluation
- Docker deployment
- PostgreSQL persistence
- REST API with Swagger UI
