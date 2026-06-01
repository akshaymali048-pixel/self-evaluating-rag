from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from self_evaluating_rag.config.logging import configure_logging
from self_evaluating_rag.config.settings import get_settings
from self_evaluating_rag.infrastructure.di.container import Container
from self_evaluating_rag.presentation.api.exception_handlers import register_exception_handlers
from self_evaluating_rag.presentation.api.middleware.timing import TimingMiddleware
from self_evaluating_rag.presentation.api.routes import documents, health, metrics, queries


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    app.state.container = Container(settings)
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description=(
            "Production RAG API with self-evaluation via RAGAS.\n\n"
            "**User Mode** — upload a document, ask a question, no ground_truth required. "
            "Returns faithfulness, answer_relevancy, and hallucination detection.\n\n"
            "**Evaluation Mode** — provide `ground_truth` with `ENABLE_REFERENCE_METRICS=true` "
            "to additionally compute context_precision and context_recall.\n\n"
            "Use `/api/v1/queries/ask-latest` or `/api/v1/queries/ask-by-filename` "
            "to avoid manual document UUIDs."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimingMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(queries.router)
    app.include_router(metrics.router)

    return app


app = create_app()
