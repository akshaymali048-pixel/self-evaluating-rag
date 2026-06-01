import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.ports.llm_service import LLMService
from self_evaluating_rag.infrastructure.llm.gemini_client import (
    create_chat_model,
    raise_if_gemini_quota_exceeded,
)
from self_evaluating_rag.infrastructure.rag.prompt_templates import RAG_ANSWER_PROMPT, RAG_SYSTEM_PROMPT


class GeminiLLMService(LLMService):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: ChatGoogleGenerativeAI | None = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = create_chat_model(self._settings, temperature=0.2)
        return self._llm

    @property
    def model_name(self) -> str:
        return self._settings.gemini_model

    async def generate_answer(self, question: str, context: str) -> str:
        prompt = RAG_ANSWER_PROMPT.format(question=question, context=context)
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = await asyncio.to_thread(self._get_llm().invoke, messages)
        except Exception as exc:
            raise_if_gemini_quota_exceeded(exc, model=self._settings.gemini_model)
            raise
        content = response.content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)
