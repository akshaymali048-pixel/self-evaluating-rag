from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from self_evaluating_rag.domain.exceptions.domain_errors import (
    DomainError,
    GeminiNotConfiguredError,
    GeminiQuotaExceededError,
    NoDocumentsError,
    OpenAINotConfiguredError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GeminiNotConfiguredError)
    async def gemini_not_configured_handler(
        _: Request, exc: GeminiNotConfiguredError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "type": exc.__class__.__name__},
        )

    @app.exception_handler(OpenAINotConfiguredError)
    async def openai_not_configured_handler(
        _: Request, exc: OpenAINotConfiguredError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "type": exc.__class__.__name__},
        )

    @app.exception_handler(GeminiQuotaExceededError)
    async def gemini_quota_exceeded_handler(
        _: Request, exc: GeminiQuotaExceededError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Gemini quota exceeded",
                "provider": exc.provider,
                "status": exc.status,
            },
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "type": exc.__class__.__name__},
        )
