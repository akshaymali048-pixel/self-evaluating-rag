from abc import ABC, abstractmethod


class LLMService(ABC):
    @abstractmethod
    async def generate_answer(self, question: str, context: str) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
