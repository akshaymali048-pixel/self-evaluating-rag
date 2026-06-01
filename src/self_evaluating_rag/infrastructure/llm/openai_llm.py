import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.ports.llm_service import LLMService
from self_evaluating_rag.infrastructure.llm.openai_client import create_chat_model
from self_evaluating_rag.infrastructure.rag.prompt_templates import RAG_ANSWER_PROMPT, RAG_SYSTEM_PROMPT


class OpenAILLMService(LLMService):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: ChatOpenAI | None = None

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = create_chat_model(self._settings, temperature=0.2)
        return self._llm

    @property
    def model_name(self) -> str:
        return self._settings.openai_model

    async def generate_answer(self, question: str, context: str) -> str:
        prompt = RAG_ANSWER_PROMPT.format(question=question, context=context)
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await asyncio.to_thread(self._get_llm().invoke, messages)
        content = response.content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)
