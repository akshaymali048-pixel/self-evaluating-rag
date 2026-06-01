from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.config.settings import Settings, get_settings
from self_evaluating_rag.presentation.api.dependencies import get_session

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
) -> dict:
    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "database": "up" if db_ok else "down",
        "llm_provider": settings.llm_provider,
        "ai_provider_configured": settings.ai_provider_configured,
        "openai_configured": settings.openai_configured,
        "gemini_configured": settings.gemini_configured,
    }
