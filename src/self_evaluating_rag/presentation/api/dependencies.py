from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.infrastructure.di.container import Container
from self_evaluating_rag.infrastructure.persistence.postgres.session import get_async_session_factory


def get_container(request: Request) -> Container:
    return request.app.state.container


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
